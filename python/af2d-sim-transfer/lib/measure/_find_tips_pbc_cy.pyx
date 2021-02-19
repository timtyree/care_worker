#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np
# Lewiner marching squares with periodic boundary conditions (pbc)
# forked Laurener marching squares without periodic boundary conditions - from https://github.com/scikit-image/scikit-image/blob/master/skimage/measure/_find_contours_cy.pyx 
# Warning: clipping instabilities may be returned when the double floating point arithmetic used here can't distinguish between vertices that should/shouldn't be mapped according to periodic boundary conditions.  I encourage you to fix this by implementing a _pbc_1d using decimal expansions/real arithmetic.  This would involve some functional programming.
cimport numpy as cnp
cnp.import_array()

cdef extern from "numpy/npy_math.h":
    bint npy_isnan(double x)

cdef inline double _get_fraction(double from_value, double to_value,
                                 double level):
    if (to_value == from_value): 
        return 0
    return ((level - from_value) / (to_value - from_value))

# cdef inline double _pbc_1d(double x_in, double x_max, double width):
#     if np.greater(x_in , x_max):
#         return x_in-width
#     return x_in

# # def inline double[:] compute_segments_in_window(ul,ll,ur,ul,level):#,segments):
# def compute_segments_in_window(double ul,double ll,double ur,double ul,double level, double rmax, double cmax, uint8_t width, uint8_t height):

#     # (double[:, :] array1, double[:, :] array2,
#     #                           double level1, double level2,
#     #                           cnp.uint8_t[:, :] mask):

#     cdef list segments = []

#     #values on the edge of square window
#     cdef tuple top, bottom, left, right

#     #values on the corners of the square window
#     cdef double ul, ur, ll, lr
#     cdef double ul2, ur2, ll2, lr2
#     cdef Py_ssize_t r0, r1, c0, c1


#     cdef unsigned char square_case = 0
#     if (ul > level): square_case += 1
#     if (ur > level): square_case += 2
#     if (ll > level): square_case += 4
#     if (lr > level): square_case += 8

#     if square_case in [0, 15]:
#         # only do anything if there's a line passing through the
#         # square. Cases 0 and 15 are entirely below/above the contour.
#         return segments
#         # continue
#     # compute the coordinates of the vertices without wrapping
#     top    = _pbc_1d(r0,rmax, width), _pbc_1d(c0 + _get_fraction(ul, ur, level) ,cmax, height) 
#     bottom = _pbc_1d(r1,rmax, width), _pbc_1d(c0 + _get_fraction(ll, lr, level) ,cmax, height) 
#     left   = _pbc_1d(r0 + _get_fraction(ul, ll, level),rmax, width), _pbc_1d(c0,cmax, height)
#     right  = _pbc_1d(r0 + _get_fraction(ur, lr, level),rmax, width), _pbc_1d(c1,cmax, height)

#     # # compute the coordinates of the vertices
#     # # always returning the bottom/right option at the boundaries.
#     # top    = _pbc_1d(r0,rmax, width), _pbc_1d(c0 + _get_fraction(ul, ur, level) ,cmax, height) 
#     # bottom = _pbc_1d(r1,rmax, width), _pbc_1d(c0 + _get_fraction(ll, lr, level) ,cmax, height) 
#     # left   = _pbc_1d(r0 + _get_fraction(ul, ll, level),rmax, width), _pbc_1d(c0,cmax, height)
#     # right  = _pbc_1d(r0 + _get_fraction(ur, lr, level),rmax, width), _pbc_1d(c1,cmax, height)

#     #wrapped boundary conditions
#     # top    = _pbc_1d(r0,rmax, width), _pbc_1d(c0 + _get_fraction(ul, ur, level) ,cmax, height) 
#     # bottom = _pbc_1d(r1,rmax, width), _pbc_1d(c0 + _get_fraction(ll, lr, level) ,cmax, height) 
#     # left   = _pbc_1d(r0 + _get_fraction(ul, ll, level),rmax, width), _pbc_1d(c0,cmax, height)
#     # right  = _pbc_1d(r0 + _get_fraction(ur, lr, level),rmax, width), _pbc_1d(c1,cmax, height)

#     if (square_case == 1):
#         # top to left
#         segments.append((top, left))
#     elif (square_case == 2):
#         # right to top
#         segments.append((right, top))
#     elif (square_case == 3):
#         # right to left
#         segments.append((right, left))
#     elif (square_case == 4):
#         # left to bottom
#         segments.append((left, bottom))
#     elif (square_case == 5):
#         # top to bottom
#         segments.append((top, bottom))
#     elif (square_case == 6):
#         # compute bilinear interpolation here as sign(face label·F(A)·(F(A)·F(C)−F(B)·F(D))) (Lewiner marching squares)
#         val = (ul - level) * (lr - level) - (ll - level) * (ur - level)
#         # TODO: check that 0 < val is used instead of 0 > val for each case (use a simple test case!) 
#         vertex_connect_high = 0 > val
#         if vertex_connect_high: #this ambiguity should be resolved with bilinear interpolation!!
#             segments.append((left, top))
#             segments.append((right, bottom))
#         else:
#             segments.append((right, top))
#             segments.append((left, bottom))
#     elif (square_case == 7):
#         # right to bottom
#         segments.append((right, bottom))
#     elif (square_case == 8):
#         # bottom to right
#         segments.append((bottom, right))
#     elif (square_case == 9):
#         # compute bilinear interpolation here as sign(face label·F(A)·(F(A)·F(C)−F(B)·F(D))) (Lewiner marching squares)
#         val = (ul - level) * (lr - level) - (ll - level) * (ur - level)
#         # TODO: check that 0 < val is used instead of 0 > val for each case (use a simple test case!) 
#         vertex_connect_high = 0 < val
#         if vertex_connect_high: #this ambiguity should be resolved with bilinear interpolation!!
#             segments.append((top, right))
#             segments.append((bottom, left))
#         else:
#             segments.append((top, left))
#             segments.append((bottom, right))
#     elif (square_case == 10):
#         # bottom to top
#         segments.append((bottom, top))
#     elif (square_case == 11):
#         # bottom to left
#         segments.append((bottom, left))
#     elif (square_case == 12):
#         # lef to right
#         segments.append((left, right))
#     elif (square_case == 13):
#         # top to right
#         segments.append((top, right))
#     elif (square_case == 14):
#         # left to top
#         segments.append((left, top))
#     return segments

def _get_intersections_pbc(double[:, :] array1, double[:, :] array2,
                          double level1, double level2,
                          cnp.uint8_t[:, :] mask):
    """Find iso-valued contours in a 2D array for a given level value.

    Iterate across the given array in a marching-squares fashion,
    looking for segments that cross 'level' for both arrays. If such a segment is
    found for both arrays, the function looks for an intersection of those two arrays.  
    If an intersection is found, the point of intersection is returned along with the interpolated value at that intersection point.
    If more than one intersection is found (2 or 4), then those intersections points are returned as a tuple.
    If no intersection points are found, [An empty tuple is returned.]
    Those coordinates are added to a growing list of intersection points, which is returned by the function.
    Positions where the boolean array ``mask`` is ``False`` are considered
    as not containing data.
    assumes that array1 and array2 have the same shape.
    

    Uses the "marching squares" method to compute a the iso-valued contours of
    the input 2D array for a particular level value. Array values are linearly
    interpolated to provide better precision for the output contours.  
    Edge cases are made unambiguous using bilinear interpolation according to [1].

     0--   1+-   2-+   3++   4--   5+-   6-+   7++
      --    --    --    --    +-    +-    +-    +-
    
     8--   9+-  10-+  11++  12--  13+-  14-+  15++
      -+    -+    -+    -+    ++    ++    ++    ++
    
    The position of the line segment that cuts through (or
    doesn't, in case 0 and 15) each square is clear, except in
    cases 6 and 9.  

    There may be multiple intersections if either segments are case 6 or 9.

    Parameters
    ----------
    array : 2D ndarray of double
        Input data in which to find contours.
    level : float
        Value along which to find contours in the array.

    .. addendum::

    The aforementioned ambiguity is resolved according to the natural reduction 
    of the method presented in [1] to two spatial dimensions.

    References
    ----------
    .. [1] Thomas Lewiner, Helio Lopes, Antonio Wilson Vieira and Geovan
           Tavares. Efficient implementation of Marching Cubes' cases with
           topological guarantees. Journal of Graphics Tools 8(2)
           pp. 1-15 (december 2003).
           :DOI:`10.1080/10867651.2003.10487582`
    """

    cdef list intersections = []

    #TODO: are these needed?
    cdef list segments1 = []
    cdef list segments2 = []

    cdef bint use_mask = mask is not None
    cdef unsigned char square_case1 = 0
    cdef unsigned char square_case2 = 0

    #values on the edge of square window
    cdef tuple top1, bottom1, left1, right1
    cdef tuple top2, bottom2, left2, right2

    #values on the corners of the square window
    cdef double ul1, ur1, ll1, lr1
    cdef double ul2, ur2, ll2, lr2
    cdef Py_ssize_t r0, r1, c0, c1
    cdef bint vertex_connect_high = False

    # #TODO: are these needed?
    # cdef bint jumped_ul = False
    # cdef bint jumped_lr = False

    #assume that array1 and array2 have the same shape
    cdef int height = array1.shape[0]
    cdef int width  = array1.shape[1]

    #precompute pbc threshold to avoid clipping resulting from floating point arithmetic
    cdef double rmax = np.around(height-0.5,2)
    cdef double cmax = np.around(width-0.5,2)

    for r0 in range(height):
        for c0 in range(width):
            # jumped_ul = False
            # jumped_lr = False

            r1, c1 = r0 + 1, c0 + 1
            if r1 >= height:
                r1 = 0
                # jumped_ul = True
            if c1 >= width:
                c1 = 0
                # jumped_lr = True

            # Skip this square if any of the four input values are masked out.
            if use_mask and not (mask[r0, c0] and mask[r0, c1] and
                                 mask[r1, c0] and mask[r1, c1]):
                continue

            #find any segments for array1
            #if none are found, continue
            ul1 = array1[r0, c0]
            ur1 = array1[r0, c1]
            ll1 = array1[r1, c0]
            lr1 = array1[r1, c1]

            square_case1 = 0
            if (ul1 > level1): square_case1 += 1
            if (ur1 > level1): square_case1 += 2
            if (ll1 > level1): square_case1 += 4
            if (lr1 > level1): square_case1 += 8

            if square_case1 in [0, 15]:
                # only do anything if there's a line passing through the
                # square. Cases 0 and 15 are entirely below/above the contour.
                continue

            #find any segments for array2
            ul2 = array2[r0, c0]
            ur2 = array2[r0, c1]
            ll2 = array2[r1, c0]
            lr2 = array2[r1, c1]

            if (ul2 > level2): square_case2 += 1
            if (ur2 > level2): square_case2 += 2
            if (ll2 > level2): square_case2 += 4
            if (lr2 > level2): square_case2 += 8

            if square_case2 in [0, 15]:
                # only do anything if there's a line passing through the
                # square. Cases 0 and 15 are entirely below/above the contour.
                continue

            #a segment exists for both contour types
            #lookup segments
            segments1 = lookup_segments(ul1,ll1,ur1,lr1,r0,r1,c0,c1,level1,square_case1)
            # segments1 = compute_segments_in_window(ul1,ll1,ur1,ul1,level1,square_case1)
            segments2 = lookup_segments(ul2,ll2,ur2,lr2,r0,r1,c0,c1,level2,square_case2)
            # if len(segments1)==0:
            #     continue
            # segments2 = compute_segments_in_window(ul2,ll2,ur2,ul2,level2,rmax,cmax,width,height)
            # if len(segments2)==0:
            #     continue
            #up to 4 iterations may occur in this nested for loop
            for u in segments1:
                for v in segments2:
                    #TODO: for each segment pair, check for an intersection point, if they exist, return the intersection ponts
                    pass

            # Skip this square if any of the four input values are NaN.
            if npy_isnan(ul1) or npy_isnan(ur1) or npy_isnan(ll1) or npy_isnan(lr1):
                continue


    # return segments
# def inline double[:] compute_segments_in_window(ul,ll,ur,ul,level):#,segments):
# # def compute_segments_in_window(double ul,double ll,double ur,double ul,double level, double rmax, double cmax, uint8_t width, uint8_t height):
# def compute_segments_in_window(double ul,double ll,double ur,
#     double ul,double level, double rmax, double cmax, uint8_t width, 
#     uint8_t height,uint8_t height,uint8_t height):

    # (double[:, :] array1, double[:, :] array2,
    #                           double level1, double level2,
    #                           cnp.uint8_t[:, :] mask):

# cdef inline tuple lookup_segments(double ul,double ll,double ur,double lr,
def lookup_segments(double ul,double ll,double ur,double lr,
    int r0, int r1,int c0, int c1,
    double level,int square_case):
    '''consider this
    segments1 = lookup_segments(ul1,ll1,ur1,lr1,r0,r1,c0,c1,level1,square_case1)'''

    cdef list segments = []

    #values on the edge of square window
    cdef tuple top, bottom, left, right

    #values on the corners of the square window
    # cdef double ul, ur, ll, lr
    # cdef double ul2, ur2, ll2, lr2
    # cdef Py_ssize_t r0, r1, c0, c1


    # cdef unsigned char square_case = 0
    # if (ul > level): square_case += 1
    # if (ur > level): square_case += 2
    # if (ll > level): square_case += 4
    # if (lr > level): square_case += 8

    # if square_case in [0, 15]:
    #     # only do anything if there's a line passing through the
    #     # square. Cases 0 and 15 are entirely below/above the contour.
    #     return segments
        # continue
    # compute the coordinates of the vertices without wrapping
    
    top    = r0, c0 + _get_fraction(ul, ur, level)
    bottom = r1, c0 + _get_fraction(ll, lr, level)
    left   = r0 + _get_fraction(ul, ll, level), c0
    right  = r0 + _get_fraction(ur, lr, level), c1

    # # compute the coordinates of the vertices
    # # always returning the bottom/right option at the boundaries.
    # top    = _pbc_1d(r0,rmax, width), _pbc_1d(c0 + _get_fraction(ul, ur, level) ,cmax, height) 
    # bottom = _pbc_1d(r1,rmax, width), _pbc_1d(c0 + _get_fraction(ll, lr, level) ,cmax, height) 
    # left   = _pbc_1d(r0 + _get_fraction(ul, ll, level),rmax, width), _pbc_1d(c0,cmax, height)
    # right  = _pbc_1d(r0 + _get_fraction(ur, lr, level),rmax, width), _pbc_1d(c1,cmax, height)

    #wrapped boundary conditions
    # top    = _pbc_1d(r0,rmax, width), _pbc_1d(c0 + _get_fraction(ul, ur, level) ,cmax, height) 
    # bottom = _pbc_1d(r1,rmax, width), _pbc_1d(c0 + _get_fraction(ll, lr, level) ,cmax, height) 
    # left   = _pbc_1d(r0 + _get_fraction(ul, ll, level),rmax, width), _pbc_1d(c0,cmax, height)
    # right  = _pbc_1d(r0 + _get_fraction(ur, lr, level),rmax, width), _pbc_1d(c1,cmax, height)

    if (square_case == 1):
        # top to left
        segments.append((top, left))
    elif (square_case == 2):
        # right to top
        segments.append((right, top))
    elif (square_case == 3):
        # right to left
        segments.append((right, left))
    elif (square_case == 4):
        # left to bottom
        segments.append((left, bottom))
    elif (square_case == 5):
        # top to bottom
        segments.append((top, bottom))
    elif (square_case == 6):
        # compute bilinear interpolation here as sign(face label·F(A)·(F(A)·F(C)−F(B)·F(D))) (Lewiner marching squares)
        val = (ul - level) * (lr - level) - (ll - level) * (ur - level)
        # TODO: check that 0 < val is used instead of 0 > val for each case (use a simple test case!) 
        vertex_connect_high = 0 > val
        if vertex_connect_high: #this ambiguity should be resolved with bilinear interpolation!!
            segments.append((left, top))
            segments.append((right, bottom))
        else:
            segments.append((right, top))
            segments.append((left, bottom))
    elif (square_case == 7):
        # right to bottom
        segments.append((right, bottom))
    elif (square_case == 8):
        # bottom to right
        segments.append((bottom, right))
    elif (square_case == 9):
        # compute bilinear interpolation here as sign(face label·F(A)·(F(A)·F(C)−F(B)·F(D))) (Lewiner marching squares)
        val = (ul - level) * (lr - level) - (ll - level) * (ur - level)
        # TODO: check that 0 < val is used instead of 0 > val for each case (use a simple test case!) 
        vertex_connect_high = 0 < val
        if vertex_connect_high: #this ambiguity should be resolved with bilinear interpolation!!
            segments.append((top, right))
            segments.append((bottom, left))
        else:
            segments.append((top, left))
            segments.append((bottom, right))
    elif (square_case == 10):
        # bottom to top
        segments.append((bottom, top))
    elif (square_case == 11):
        # bottom to left
        segments.append((bottom, left))
    elif (square_case == 12):
        # lef to right
        segments.append((left, right))
    elif (square_case == 13):
        # top to right
        segments.append((top, right))
    elif (square_case == 14):
        # left to top
        segments.append((left, top))
    return segments