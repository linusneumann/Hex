# Part of the code was used from https://youtu.be/pWz2scRHCI8?si=AXTAjDH0GcVr2W4R (accessed: 22.05.2025) 
class DisjointSet:
    def __init__(self, elems):
        self.elems = elems
        self.parent = {}
        self.size = {}
        self.history = []
        for elem in elems:
            self.make_set(elem)

    def make_set(self, x):
        self.parent[x] = x
        self.size[x] = 1

    def find(self, x):
        if self.parent[x] == x:
            return x
        else:
            self.parent[x] = self.find(self.parent[x])
            return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return
        
        self.history.append((root_x, root_y, self.size[root_x], self.size[root_y]))
        #print(f"union : {root_x}, {root_y}")

        if self.size[root_x] < self.size[root_y]:
            self.parent[root_x] = root_y
            self.size[root_y] += self.size[root_x]
        else:
            self.parent[root_y] = root_x
            self.size[root_x] += self.size[root_y]

    def undo_union(self):
        if not self.history:
            #print("Kein Union-Vorgang zum Rückgängigmachen.")
            return
        
        root_x, root_y, size_x, size_y = self.history.pop()
        #print(f"undoing:{root_x},{root_y}")
        
        # Wiederherstellen der vorherigen Zustände
        self.parent[root_x] = root_x
        self.parent[root_y] = root_y
        self.size[root_x] = size_x
        self.size[root_y] = size_y
    
    def add_elements(self,elems):
        for elem in elems:
            if elem not in self.elems:
                self.elems.append(elem)
                self.make_set(elem)
       
if __name__ == "__main__":
    ds = DisjointSet([1, 2, 3, 4])

    ds.union(1, 2)
    ds.union(3, 4)
    ds.union(2, 3)

    print(ds.find(1))  # sollte die gleiche wurzel wie 3 haben
    print(ds.find(4))  # sollte auch zur gleichen menge gehören

    ds.undo_union()  # macht (2,3) rückgängig
    print(ds.find(1), ds.find(3))  # sollten nun getrennte mengen sein

    ds.undo_union()  # macht (3,4) rückgängig
    print(ds.find(3), ds.find(4))  # sollten nun wieder getrennte mengen sein