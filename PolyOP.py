"""data structure of the project

1.Rectangle(Rect):save down-left and up-right point
2.Rectilinear Polygon(Poly):a extension of Rectangle which vertices is more than 4
3.Graph(Graph):change Rectilinear Polygon into a better data structure, then merge and clip is easier to perform
"""
import sys
import numpy as np

class Rect:
    def __init__(self):
        self.x = []
        self.y = []

    def set_ver(self, x1, y1, x2, y2):
        self.x.append(x1)
        self.y.append(y1)
        self.x.append(x2)
        self.y.append(y2)


class Polygon:
    def __init__(self):
        self.x = []
        self.y = []

    def addv(self, x_coord, y_coord):
        self.x.append(x_coord)
        self.y.append(y_coord)

class Graph:
    def __init__(self, Pol):
        self.x = sorted(list( set( Pol.x )))
        self.y = sorted(list( set( Pol.y )))
        self.Pol = Pol
        self.vertical = [[0]*(self.ver_len()-1) for i in range(self.hor_len())]
        self.horizontal = [[0] * self.ver_len() for i in range(self.hor_len()-1)]
        self.mat = [[0]*(self.ver_len()-1) for i in range(self.hor_len()-1)]


    def hor_len(self):
        return len(self.x)

    def ver_len(self):
        return len(self.y)

    "can use if Pol exists"
    def calculate_v(self):
        temp=[[0]*(self.ver_len()-1) for i in range(self.hor_len())]
        for i in range(len(self.Pol.x)-1):
            if(self.Pol.x[i] == self.Pol.x[i+1]):
                if(self.Pol.y[i] > self.Pol.y[i+1]):
                    j1=self.y.index(self.Pol.y[i+1])
                    j2=self.y.index(self.Pol.y[i])
                    for j in range(j1, j2):
                        temp[self.x.index(self.Pol.x[i])][j] = 1
                elif (self.Pol.y[i] < self.Pol.y[i+1]):
                    j1=self.y.index(self.Pol.y[i])
                    j2=self.y.index(self.Pol.y[i+1])
                    for j in range(j1, j2):
                        temp[self.x.index(self.Pol.x[i])][j] = 1

        return temp


    def calculate_h(self):
        temp = [[0] * self.ver_len() for i in range(self.hor_len()-1)]
        for i in range(len(self.Pol.x)-1):
            if(self.Pol.y[i] == self.Pol.y[i+1]):
                if(self.Pol.x[i] > self.Pol.x[i+1]):
                    j1=self.x.index(self.Pol.x[i+1])
                    j2=self.x.index(self.Pol.x[i])
                    for j in range(j1, j2):
                        temp[j][self.y.index(self.Pol.y[i])]=1
                elif (self.Pol.x[i] < self.Pol.x[i+1]):
                    j1=self.x.index(self.Pol.x[i])
                    j2=self.x.index(self.Pol.x[i+1])
                    for j in range(j1, j2):
                        temp[j][self.y.index(self.Pol.y[i])]=1

        return temp

    def occupied_mat(self,ver):

        temp=[[0]*(self.ver_len()-1) for i in range(self.hor_len()-1)]
        for i in range(self.hor_len()-1):
            for j in range(self.ver_len()-1):
                count = 0
                flag = i

                while flag >= 0:
                    count += ver[flag][j]
                    flag -= 1

                if count%2!=0:
                    temp[i][j]=1
        return temp

    """use this function to check mapping of polygon is correct or not"""

    def draw(self):
        temp = self.occupied_mat(self.calculate_v())
        for j in range(self.ver_len()-2,-1,-1):
            for i in range(self.hor_len()-1):
                if i==self.hor_len()-2:
                    print(temp[i][j])
                else:
                    print(temp[i][j], end=' ')

def out_draw(matrix):
    len_x=len(matrix)
    len_y=len(matrix[0])
    for j in range(len_y-1,-1,-1):
        for i in range(len_x):
            if i==len_x-1:
                print(matrix[i][j])
            else:
                print(matrix[i][j], end=' ')



def Merge(G1,G2,ver1):
    "handle fist merge"
    if G1.x==[] and ver1==[]:
        G2.vertical=G2.calculate_v()
        G2.mat=G2.occupied_mat(G2.vertical)
        return G2
    temp = Graph(Polygon())
    x_coord = sorted(list(set(G1.x+G2.x)))
    y_coord = sorted(list(set(G1.y+G2.y)))
    graph1=G1.occupied_mat(ver1)
    graph2=G2.occupied_mat(G2.calculate_v())

    temp.x = x_coord
    temp.y = y_coord
    graph = [[0]*(temp.ver_len()-1) for i in range(temp.hor_len()-1)]
    for i in range(len(x_coord)-1):
        for j in range(len(y_coord)-1):
            x1=find_cloesest_smaller(x_coord[i],G1.x)
            x2=find_cloesest_smaller(x_coord[i],G2.x)
            y1=find_cloesest_smaller(y_coord[j],G1.y)
            y2=find_cloesest_smaller(y_coord[j],G2.y)

            if x1==-1 or x1==len(G1.x)-1 or y1==-1 or y1==len(G1.y)-1:
                if x2==-1 or x2==len(G2.x)-1 or y2==-1 or y2==len(G2.y)-1:
                    pass
                else:
                    if graph2[x2][y2]==1:
                        graph[i][j]=1
            else:
                if x2 == -1 or x2 == len(G2.x) - 1 or y2 == -1 or y2 == len(G2.y) - 1:

                    if graph1[x1][y1]==1:
                        graph[i][j]=1
                else:
                    if graph1[x1][y1]==1 or graph2[x2][y2]==1:
                        graph[i][j]=1

    vertical=[[0]*(len(temp.y)-1) for i in range(len(temp.x))]
    for i in range(len(temp.x)-1):
        for j in range(len(temp.y)-1):
            if graph[i][j]==1:
                vertical[i][j]=(vertical[i][j]+1)%2
                vertical[i+1][j]=(vertical[i+1][j]+1)%2

    temp.vertical=vertical
    temp.mat=graph
    return thin(temp)

def Clip(G1,G2):
    temp=Graph(Polygon())
    tempx=sorted(list(set(G1.x+G2.x)))
    tempy=sorted(list(set(G1.y+G2.y)))
    graph1=G1.mat
    graph2=G2.occupied_mat(G2.calculate_v())
    x_left=tempx.index(G1.x[0])
    x_right=tempx.index(G1.x[-1])
    y_low=tempy.index(G1.y[0])
    y_high=tempy.index(G1.y[-1])
    temp.x=tempx[x_left:x_right+1]
    temp.y=tempy[y_low:y_high+1]

    graph = [[0] * (temp.ver_len() - 1) for i in range(temp.hor_len() - 1)]
    for i in range(len(temp.x)-1):
        for j in range(len(temp.y)-1):
            x1=find_cloesest_smaller(temp.x[i],G1.x)
            x2=find_cloesest_smaller(temp.x[i],G2.x)
            y1=find_cloesest_smaller(temp.y[j],G1.y)
            y2=find_cloesest_smaller(temp.y[j],G2.y)
            if x1==-1 or x1==len(G1.x)-1 or y1==-1 or y1==len(G1.y)-1:
                if x2==-1 or x2==len(G2.x)-1 or y2==-1 or y2==len(G2.y)-1:
                    pass
                else:
                    if graph2[x2][y2]==1:
                        graph[i][j]=0
            else:
                if x2==-1 or x2==len(G2.x)-1 or y2==-1 or y2==len(G2.y)-1:
                    if graph1[x1][y1]==1:
                        graph[i][j]=1
                else:
                    if graph1[x1][y1]==1 and graph2[x2][y2]==0:
                        graph[i][j]=1
    vertical=[[0]*(len(temp.y)-1) for i in range(len(temp.x))]
    for i in range(len(temp.x)-1):
        for j in range(len(temp.y)-1):
            if graph[i][j]==1:
                vertical[i][j]=(vertical[i][j]+1)%2
                vertical[i+1][j]=(vertical[i+1][j]+1)%2
    temp.vertical=vertical
    temp.mat=graph
    return thin(temp)



def find_cloesest_smaller(flag, sorted_array):
    i=0
    if flag<sorted_array[i]:
        return -1
    if flag>=sorted_array[len(sorted_array)-1]:
        return len(sorted_array)-1
    while(flag>=sorted_array[i]):
        i+=1
    i-=1
    return i



def thin(graph):
    matrix=graph.mat
    x_nums=graph.x
    y_nums=graph.y
    prune_x=[]
    for i in range(len(matrix)-1):
        if matrix[i]==matrix[i+1]:
            prune_x.append(x_nums[i+1])
    for element in prune_x:
        index=x_nums.index(element)
        del x_nums[index]
        del matrix[index]

    matrix_T=np.array(matrix).T.tolist()
    prune_y=[]

    for j in range(len(matrix_T)-1):
        if matrix_T[j]==matrix_T[j+1]:
            prune_y.append(y_nums[j+1])
    for element in prune_y:
        index=y_nums.index(element)
        del y_nums[index]
        del matrix_T[index]
    matrix=np.array(matrix_T).T.tolist()

    vertical=[[0]*(len(y_nums)-1) for i in range(len(x_nums))]
    for i in range(len(x_nums)-1):
        for j in range(len(y_nums)-1):
            if matrix[i][j]==1:
                vertical[i][j]=(vertical[i][j]+1)%2
                vertical[i+1][j]=(vertical[i+1][j]+1)%2
    graph=Graph(Polygon())
    graph.x=x_nums
    graph.y=y_nums
    graph.vertical=vertical
    graph.mat=matrix
    return graph






"small solve method"
def solve(start_x,stop_x,dict,mat,x_nums,y_nums):
    temp_y=0
    for i in range(len(y_nums)-2):
        if mat[start_x][i]==0 and mat[start_x][i+1]==1:
            temp_y=i+1

        if mat[start_x][i]==1 and mat[start_x][i+1]==0:
            temp_Rect=Rect()
            temp_Rect.set_ver(x_nums[start_x],y_nums[temp_y],x_nums[stop_x],y_nums[i+1])
            check_tuple=(x_nums[start_x],y_nums[i+1])
            "Merge Method"
            if check_tuple in dict and dict[check_tuple].y[0]==y_nums[temp_y]:
                temp2_Rect=dict[check_tuple]
                del dict[check_tuple]
                merge_Rect=Rect()
                merge_Rect.set_ver(temp2_Rect.x[0],temp2_Rect.y[0],x_nums[stop_x],y_nums[i+1])
                dict[(x_nums[stop_x],y_nums[i+1])]=merge_Rect
            else:
                dict[(x_nums[stop_x],y_nums[i+1])]=temp_Rect

    if mat[start_x][-1]==1:
        temp_Rect=Rect()
        temp_Rect.set_ver(x_nums[start_x],y_nums[temp_y],x_nums[stop_x],y_nums[-1])
        check_tuple=(x_nums[start_x],y_nums[-1])
        if check_tuple in dict and dict[check_tuple].y[0] == y_nums[temp_y]:
            temp2_Rect = dict[check_tuple]
            del dict[check_tuple]
            merge_Rect = Rect()
            merge_Rect.set_ver(temp2_Rect.x[0], temp2_Rect.y[0], x_nums[stop_x], y_nums[-1])
            dict[(x_nums[stop_x], y_nums[-1])] = merge_Rect
        else:
            dict[(x_nums[stop_x], y_nums[-1])] = temp_Rect
    return dict
"""slice horizontally:
    """
def slice_horizontally(mat,Poly_x,Poly_y):
    if len(mat)!=len(Poly_x)-1 or len(mat[0])!=len(Poly_y)-1:
        return {}
    cuts_index=[]
    mat_T=np.array(mat).T.tolist()
    Rect_dict={}

    for j in range(len(mat_T)-1):
        if mat_T[j]!=mat_T[j+1]:
            cuts_index.append(j)
    if len(Poly_x)>=2:
        if len(cuts_index)==0:
            start_y=0
            stop_y=len(Poly_y)-1
            Rect_dict=solve(start_y,stop_y,Rect_dict,mat_T,Poly_y,Poly_x)
        else:
            for y_index in range(len(cuts_index)):
                if y_index==0:
                    start_y=0
                    stop_y=cuts_index[y_index]+1
                    Rect_dict=solve(start_y,stop_y,Rect_dict,mat_T,Poly_y,Poly_x)
                else:
                    start_y=cuts_index[y_index-1]+1
                    stop_y=cuts_index[y_index]+1
                    Rect_dict=solve(start_y,stop_y,Rect_dict,mat_T,Poly_y,Poly_x)
                if cuts_index[y_index]==cuts_index[-1]:
                    start_y=cuts_index[y_index]+1
                    stop_y=len(Poly_y)-1
                    Rect_dict=solve(start_y,stop_y,Rect_dict,mat_T,Poly_y,Poly_x)
    tuple_list=[]
    Rect_list=[]
    for key in Rect_dict:
        tuple_list.append(key)
        Rect_list.append(Rect_dict[key])
    del Rect_dict
    Result={}
    for i in range(len(tuple_list)):
        (y,x)=tuple_list[i]
        add_Rect=Rect()
        add_Rect.y=Rect_list[i].x
        add_Rect.x=Rect_list[i].y
        Result[(x,y)]=add_Rect
    return  Result

"""slice vertically:
        input is a 2-D array, where output is the array of sliced rectangle"""
def slice_vertically(mat,Poly_x,Poly_y):
    if len(mat)!=len(Poly_x)-1 or len(mat[0])!=len(Poly_y)-1:
        return {}
    cuts_index=[]
    Rect_dict={}

    for i in range(len(mat)-1):
        current=mat[i]
        next=mat[i+1]
        if current!=next:
            cuts_index.append(i)

    if len(Poly_y)>=2:
        if len(cuts_index)==0:
            start_x=0
            stop_x=len(Poly_x)-1
            Rect_dict=solve(start_x,stop_x,Rect_dict,mat,Poly_x,Poly_y)
        else:
            for x_index in range(len(cuts_index)):
                if x_index==0:
                    start_x=0
                    stop_x=cuts_index[x_index]+1
                    Rect_dict=solve(start_x,stop_x,Rect_dict,mat,Poly_x,Poly_y)
                else:
                    start_x=cuts_index[x_index-1]+1
                    stop_x=cuts_index[x_index]+1
                    Rect_dict=solve(start_x,stop_x,Rect_dict,mat,Poly_x,Poly_y)
                if cuts_index[x_index]==cuts_index[-1]:
                    start_x=cuts_index[x_index]+1
                    stop_x=len(Poly_x)-1
                    Rect_dict=solve(start_x,stop_x,Rect_dict,mat,Poly_x,Poly_y)
    return Rect_dict



def main():

    operation=[]
    DATA=[]
    "Data parsing using command  python3 PolyOP.py 檔名 eg.OpenCase_1.txt"
    fp = open(sys.argv[1], "r")
    lines=fp.readlines()
    fp.close()
    temp=lines[0].split(sep=' ')
    for j in range(1, (len(temp) - 1)):
        operation.append(temp[j])
    flag=False
    for i in range(1,(len(lines))):
        temp = lines[i].split(sep=' ')
        if temp[0]=='DATA':
            flag=True
            DATA.append([temp[2]])
        if temp[0]=='\n' or temp[0]=='END':
            flag=False
        if flag==True and temp[0]!='DATA':
            Pol = Polygon()
            length = int(len(temp)/2-1)
            for k in range(length):
                Pol.addv(int(temp[2*k+1]),int(temp[2*k+2]))
            DATA[-1].append(Pol)

    "#operations , # Polygons in each operation and #lines"
    print(operation)
    print(len(lines))
    Result_Graph=Graph(Polygon())

    result={}
    for command_id in range(len(operation)):
        if operation[command_id][0]=='M':
            for loop in range(len(DATA)):
                if DATA[loop][0] ==operation[command_id]:
                    index=loop

            for i in range(1,len(DATA[index])):
                print(operation[command_id], end=' ')
                print(i, end=' ')
                print('/',end=' ')
                print(len(DATA[index])-1)
                temp_Pol=DATA[index][i]
                temp_Graph=Graph(temp_Pol)
                Result_Graph=Merge(Result_Graph,temp_Graph,Result_Graph.vertical)

                if i==len(DATA[index])-1:

                    print("Merge_x is done")

        if operation[command_id][0]=='C':
            
            for loop in range(len(DATA)):
                if DATA[loop][0] ==operation[command_id]:
                    index=loop

            for i in range(1,len(DATA[index])):
                print(operation[command_id],end=' ')
                print(i, end=' ')
                print('/', end=' ')
                print(len(DATA[index]) - 1)
                temp_Pol = DATA[index][i]
                temp_Graph = Graph(temp_Pol)
                Result_Graph=Clip(Result_Graph,temp_Graph)
                'print(i)'
                if i==len(DATA[index])-1:
                    print("Clip_x is done")
        if operation[command_id]=="SV":
            result=slice_vertically(Result_Graph.mat,Result_Graph.x,Result_Graph.y)

        if operation[command_id]=="SH":
            result=slice_horizontally(Result_Graph.mat,Result_Graph.x,Result_Graph.y)


    fo=open("result.txt","w")
    for key in result:
        s="RECT "
        s+=str(result[key].x[0])
        s+=' '
        s+=str(result[key].y[0])
        s+=' '
        s+=str(result[key].x[1])
        s+=' '
        s+=str(result[key].y[1])
        s+=';\n'
        fo.write(s)
    fo.close()

if __name__ == '__main__':
    main()
