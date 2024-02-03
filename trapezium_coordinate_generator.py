import math
def trapezium_coordinates(list_coordinates):#list_coordinates=[[x1,y1](top_right),[x2,y2](bottom_left),[x3,y3](bottom_right)]
    dis=math.sqrt(((list_coordinates[1][0]-list_coordinates[2][0])**2)+((list_coordinates[1][1]-list_coordinates[2][1])**2))
    dis=math.floor(dis)
    print(dis)
    x3=list_coordinates[1][0]+dis
    y3=list_coordinates[1][1]
    list_coordinates[2]=[x3,y3]
    asq=(list_coordinates[0][0]-list_coordinates[1][0])**2+(list_coordinates[0][1]-list_coordinates[1][1])**2
    csq=(list_coordinates[0][0]-list_coordinates[2][0])**2+(list_coordinates[0][1]-list_coordinates[2][1])**2
    short_side=dis-(dis*dis+csq-asq)/dis
    print(short_side)
    short_side=math.floor(short_side)
    x4=list_coordinates[0][0]-short_side
    y4=list_coordinates[0][1]
    print(x4,y4)
    list_coordinates.append([x4,y4])
    return list_coordinates


if __name__ == "__main__":
    list_coordinates=[[5,4],[0,0],[9,1]]
    print(trapezium_coordinates(list_coordinates))
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    # Coordinates of the trapezium
    coordinates = list_coordinates
    k=coordinates[0]
    coordinates[0]=coordinates[3]
    coordinates[3]=k

    # Separate x and y coordinates
    x_coords, y_coords = zip(*coordinates)

    # Plot the trapezium
    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='datalim')
    ax.add_patch(Polygon(coordinates, closed=True, edgecolor='b', facecolor='lightgray'))

    # Set axis limits
    ax.set_xlim(min(x_coords) - 1, max(x_coords) + 1)
    ax.set_ylim(min(y_coords) - 1, max(y_coords) + 1)

    # Show the plot
    plt.grid(True)
    plt.title('Trapezium Plot')
    plt.show()


