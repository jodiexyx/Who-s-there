class Tracker:

    def __init__(self,tolerance):
        self.last_faces = []
        self.tolerance = tolerance**2

    def update(self,faces):
        ret = []
        for face in faces:
            #print('face detected')
            recognize = True
            for last in self.last_faces:
                if self._is_same(face,last):
                    recognize = False
            if recognize == True:
                print('requesting recognition')
                ret.append(face)

        self.last_faces = faces

        return ret

    def _is_same(self,a,b):
        xa,ya,wa,ha = a
        xb,yb,wb,hb = b
        
        if (xa-xb)**2 + (ya-yb)**2 <= self.tolerance:
            return True
        else:
            return False
        
