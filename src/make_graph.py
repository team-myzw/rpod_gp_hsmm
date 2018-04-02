# encoding: utf8
from __future__ import unicode_literals
import numpy
import pylab
import matplotlib
import random
import os
"""
colors = [
'aliceblue',
'antiquewhite',
'aqua',
'aquamarine',
'azure',
'beige',
'bisque',
'black',
'blanchedalmond',
'blue',
'blueviolet',
'brown',
'burlywood',
'cadetblue',
'chartreuse',
'chocolate',
'coral',
'cornflowerblue',
'cornsilk',
'crimson',
'cyan',
'darkblue',
'darkcyan',
'darkgoldenrod',
'darkgray',
'darkgreen',
'darkkhaki',
'darkmagenta',
'darkolivegreen',
'darkorange',
'darkorchid',
'darkred',
'darksalmon',
'darkseagreen',
'darkslateblue',
'darkslategray',
'darkturquoise',
'darkviolet',
'deeppink',
'deepskyblue',
'dimgray',
'dodgerblue',
'firebrick',
'floralwhite',
'forestgreen',
'fuchsia',
'gainsboro',
'ghostwhite',
'gold',
'goldenrod',
'gray',
'green',
'greenyellow',
'honeydew',
'hotpink',
'indianred',
'indigo',
'ivory',
'khaki',
'lavender',
'lavenderblush',
'lawngreen',
'lemonchiffon',
'lightblue',
'lightcoral',
'lightcyan',
'lightgoldenrodyellow',
'lightgreen',
'lightgray',
'lightpink',
'lightsalmon',
'lightseagreen',
'lightskyblue',
'lightslategray',
'lightsteelblue',
'lightyellow',
'lime',
'limegreen',
'linen',
'magenta',
'maroon',
'mediumaquamarine',
'mediumblue',
'mediumorchid',
'mediumpurple',
'mediumseagreen',
'mediumslateblue',
'mediumspringgreen',
'mediumturquoise',
'mediumvioletred',
'midnightblue',
'mintcream',
'mistyrose',
'moccasin',
'navajowhite',
'navy',
'oldlace',
'olive',
'olivedrab',
'orange',
'orangered',
'orchid',
'palegoldenrod',
'palegreen',
'paleturquoise',
'palevioletred',
'papayawhip',
'peachpuff',
'peru',
'pink',
'plum',
'powderblue',
'purple',
'red',
'rosybrown',
'royalblue',
'saddlebrown',
'salmon',
'sandybrown',
'seagreen',
'seashell',
'sienna',
'silver',
'skyblue',
'slateblue',
'slategray',
'snow',
'springgreen',
'steelblue',
'tan',
'teal',
'thistle',
'tomato',
'turquoise',
'violet',
'wheat',
'white',
'whitesmoke',
'yellow',
'yellowgreen'
]
"""

colors = [
"r" , "g", "b", "c", "m", "y", "k" , "w",
"pink", "navy", "yellowgreen", "silver", "orange", "indigo", "gray", "darkmagenta"
]


def make_graph( dirname ):
    for d in range(4):
        res = numpy.loadtxt( os.path.join(dirname, "segm%03d.txt" % d ) )
        correct = numpy.loadtxt("correct%03d.txt" % d)
        pylab.clf()

        pylab.subplot( 2,1,1 )
        for x,c in enumerate(res):
            c = int(c)
            print c,
            pylab.fill_between( [x,x+1], [0,0], [1,1], facecolor=colors[c], edgecolor="none")
        pylab.xlim(0,len(res))
        pylab.xticks([])
        pylab.yticks([])

        pylab.subplot( 2,1,2 )
        for x,c in enumerate(correct):
            c = int(c)
            print c,
            pylab.fill_between( [x,x+1], [0,0], [1,1], facecolor=colors[c], edgecolor="none")
        pylab.xlim(0,len(correct))
        pylab.xticks([])
        pylab.yticks([])


        pylab.savefig( os.path.join( dirname, "result%d.svg" % d ) )
        pylab.savefig( os.path.join( dirname, "result%d.png" % d ) )


def main():
    make_graph( "learn" )



if __name__ == '__main__':
    main()