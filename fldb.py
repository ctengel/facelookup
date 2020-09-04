"""Face Lookup DataBase

A wrapper around semi-stateless facial recognition software with no ability to track people
Stores Images sent in, Faces returned, and allows associating Persons
"""


class FLDB:
    """Face Lookup DataBase itself - interacts with Rekognition and stores other objects"""
    def __init__(self, rekog_client, rekog_coll, bucket, create=False, initial_data=None):
        self.rekog = rekog_client
        self.coll = rekog_coll
        self.bucket = bucket
        if create:
            self.create_all()
        self.persons = {} # persons by name
        self.images = {} # images by external/filename
        self.faces = {} # faces by faceID
        if initial_data:
            self.import_db(initial_data)
    def add_person(self, name):
        """Add a new person to DB and return object so faces can be associated"""
        assert name not in self.persons
        person = Person(name)
        self.persons[name] = person
        return person
    def get_person(self, name, create=False):
        """Get a given person from DB"""
        if create and name not in self.persons:
            return self.add_person(name)
        return self.persons[name]
    def add_image(self, url, index=True):
        """Add an image and do some face recognition, returning an Image object with Faces"""
        assert url not in self.images
        image = Image(self, url)
        self.images[url] = image
        if index:
            # TODO refactor into add_face
            self.faces.update({x.fid: x for x in image.get_faces()})
        return image
    def get_image(self, url, create=False):
        """Get an Image with given path"""
        if create and url not in self.images:
            # NOTE we don't index if its just auto-created object
            return self.add_image(url, index=False)
        return self.images[url]
    def get_face(self, fid=None, data=None, create=True):
        """Get a Face with given data"""
        if not fid:
            fid = data['FaceId']
        if create and data and fid not in self.faces:
            # TODO refactor into add_face
            self.faces[fid] = Face(self, data)
        return self.faces[fid]
    def list_faces(self, refresh=False):
        """List all known face IDs"""
        if refresh:
            self.faces = {x['FaceId']: Face(self, x)
                          for x in self.rekog.list_faces(CollectionId=self.coll)}
            # TODO handle NextToken
        return self.faces.keys()
    def list_images(self):
        """List all known image paths"""
        return self.images.keys()
    def list_persons(self):
        """List all known person names"""
        return self.persons.keys()
    def create_all(self):
        """Initialize the collection in the recognition backend"""
        return self.rekog.create_collection(CollectionId=self.coll)['CollectionArn']
    def delete_all(self):
        """Clear state from the backend"""
        self.rekog.delete_collection(CollectionId=self.coll)
    def export_db(self):
        # TODO do it
        pass
    def import_db(self, initial_data):
        # TODO do it
        pass
    def _index_faces(self, url):
        """Call backend API on given URL"""
        # NOTE uses name as ID and ignores imageID
        # TODO handle Orientation or Unindexed
        return self.rekog.index_faces(CollectionId=self.coll,
                                      Image={'S3Object': {'Bucket': self.bucket, 'Name': url}},
                                      DetectionAttributes=['ALL'],
                                      ExtermalImageId=url)['FaceRecords']
    def index_faces(self, image):
        # NOTE this may be tricky if imageid needed from image!
        return [Face(self, x) for x in self._index_faces(image.url)]
    def _search_faces(self, fid):
        """Call backend API on given ID"""
        return self.rekog.search_faces(CollectionId=self.coll, FaceId=fid)['FaceMatches']
    def search_faces(self, face):
        return [(self.get_face(data=x['Face'], create=True), x['Similarity']/100.0)
                for x in self._search_faces(face.fid)]

class Image:
    """An image which is already in S3"""
    def __init__(self, db, url):
        # NOTE allow more sophisticated external ID or even rekog image ID
        self.url = url
        self.db = db
        self.faces = None
    def get_url(self):
        return self.url
    def get_faces(self):
        if self.faces:
            return self.faces
        self.faces = self.db.index_faces(self)
        return self.faces


class Face:
    # TODO merge two seperate ways Faces get created internally
    #      one is FLDB.add_image()->Image.get_faces()
    #      another is Person.get_faces()->Face.get_similar()
    """A face already indexed"""
    def __init__(self, db, fulldata, image=None):
        # NOTE we don't allow ID only at this point since no way to get it off DB
        self.fid = fulldata['Face']['FaceId']
        self.fulldata = fulldata
        self.db = db
        if image:
            self.image = image
        else:
            self.image = self.db.get_image(self.fulldata['Face']['ExternalImageId'], create=True)
        self.similar = None
        self.person = None
    def get_similar(self, refresh=False):
        """Return tuples of Face, Similarity (0.0-0.1)"""
        if refresh or self.similar is None:
            self.similar = self.db.search_faces(self)
        return self.similar
    def get_image(self):
        """Return associated Image"""
        return self.image
    def _set_person(self, person):
        self.person = person
    def set_person(self, person):
        """Confirm that this image is of a Person"""
        self._set_person(person)
        person._add_face(self)
    def get_person(self):
        if self.person:
            return [(self.person, 1)]
        return [(x[0].person, x[1]) for x in self.get_similar() if x[0].persom]

class Person:
    """A person"""
    def __init__(self, name):
        # NOTE we will need to evolve to a better ID system
        self.name = name
        self.faces = []
    def _add_face(self, face):
        self.faces.append(face)
    def add_face(self, face):
        """Confirm that given face is of this person"""
        self._add_face(face)
        face._set_person(self)
    def get_faces(self, guess=True, refresh=False):
        faces = {}
        if guess:
            for confirmed in self.faces:
                # get only similar faces without persons (ours added below)
                similar = [x for x in confirmed.get_similar(refresh=refresh) if not x.person]
                for sim in similar:
                    # get current value or 0; then increment by new
                    old = faces.get(sim[0].fid, (sim[0], 0.0))
                    faces[sim.fid] = (old[0], old[1] + sim[1])
            for thisguess in faces.values():
                thisguess = (thisguess[0], thisguess[1] / float(len(self.faces)))
        # add in known as 100%
        faces.update({x.fid: (x, 1.0) for x in self.faces})
        # sort with the best guesses first
        return sorted(faces.values(), key=lambda x: x[1], reverse=True)
