"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

"""
N-D Bresenham line algo
"""
import numpy as np
def _bresenhamline_nslope(slope):
    """
    Normalize slope for Bresenham's line algorithm.

    >>> s = np.array([[-2, -2, -2, 0]])
    >>> _bresenhamline_nslope(s)
    array([[-1., -1., -1.,  0.]])

    >>> s = np.array([[0, 0, 0, 0]])
    >>> _bresenhamline_nslope(s)
    array([[ 0.,  0.,  0.,  0.]])

    >>> s = np.array([[0, 0, 9, 0]])
    >>> _bresenhamline_nslope(s)
    array([[ 0.,  0.,  1.,  0.]])
    """
    scale = np.amax(np.abs(slope), axis=1).reshape(-1, 1)
    zeroslope = (scale == 0).all(1)
    scale[zeroslope] = np.ones(1)
    normalizedslope = np.array(slope, dtype=np.double) / scale
    normalizedslope[zeroslope] = np.zeros(slope[0].shape)
    return normalizedslope

def _bresenhamlines(start, end, max_iter):
    """
    Returns npts lines of length max_iter each. (npts x max_iter x dimension) 

    >>> s = np.array([[3, 1, 9, 0],[0, 0, 3, 0]])
    >>> _bresenhamlines(s, np.zeros(s.shape[1]), max_iter=-1)
    array([[[ 3,  1,  8,  0],
            [ 2,  1,  7,  0],
            [ 2,  1,  6,  0],
            [ 2,  1,  5,  0],
            [ 1,  0,  4,  0],
            [ 1,  0,  3,  0],
            [ 1,  0,  2,  0],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  0]],
    <BLANKLINE>
           [[ 0,  0,  2,  0],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  0],
            [ 0,  0, -1,  0],
            [ 0,  0, -2,  0],
            [ 0,  0, -3,  0],
            [ 0,  0, -4,  0],
            [ 0,  0, -5,  0],
            [ 0,  0, -6,  0]]])
    """
    if max_iter == -1:
        max_iter = np.amax(np.amax(np.abs(end - start), axis=1))
    npts, dim = start.shape
    nslope = _bresenhamline_nslope(end - start)

    # steps to iterate on
    stepseq = np.arange(1, max_iter + 1)
    stepmat = np.tile(stepseq, (dim, 1)).T

    # some hacks for broadcasting properly
    bline = start[:, np.newaxis, :] + nslope[:, np.newaxis, :] * stepmat

    # Approximate to nearest int
    return np.array(np.rint(bline), dtype=start.dtype)

def draw_gradient_line(canvas, lines, tint_start, tint_end):
    """Draws a gradient line on the canvas.

    Args:
        canvas: The canvas to draw on.
        lines: The points of the line.
        tint_start: The color at the start of the line.
        tint_end: The color at the end of the line.
    """
    line_len = len(lines)
    for j, p in enumerate(lines):
        line_len = len(lines)
        u, v = p
        u = np.clip(u, 0, canvas.shape[1] - 1)
        v = np.clip(v, 0, canvas.shape[0] - 1)
        t = j / (line_len-1) if line_len > 1 else 1.0
        color = ((1-t)*tint_start + t*tint_end)
        canvas[int(v), int(u)] = (color).astype(np.uint8)

def bresenham_thick_line(canvas, x1, y1, x2, y2, thickness, tint_start, tint_end):
    """Draws a thick line using Bresenham's algorithm.

    Args:
        x1: X-coordinate of the starting point.
        y1: Y-coordinate of the starting point.
        x2: X-coordinate of the ending point.
        y2: Y-coordinate of the ending point.
        thickness: Thickness of the line.
    """

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = (dx if dx > dy else -dy) / 2

    for i in range(thickness // 2 + 1):
        x, y = x1, y1
        
        lines = []
        while True:
            # plot(canvas, x, y)  # Replace with your plotting function
            lines.append((x, y))
            if x == x2 and y == y2:
                break
            e2 = err
            if e2 > -dx:
                err -= dy
                x += sx
            if e2 < dy:
                err += dx
                y += sy

        
        draw_gradient_line(canvas, lines, tint_start, tint_end)
        
        #Calculate offset for next parallel line
        if dx > dy:
            x1, y1 = x1, y1 + 1
            x2, y2 = x2, y2 + 1
        else:
            x1, y1 = x1 + 1, y1
            x2, y2 = x2 + 1, y2
        err = (dx if dx > dy else -dy) / 2 # Reset error for next line
    
    for i in range(thickness // 2):
        x, y = x1, y1
        lines = []
        while True:
            # plot(canvas, x, y)  # Replace with your plotting function
            lines.append((x, y))
            if x == x2 and y == y2:
                break
            e2 = err
            if e2 > -dx:
                err -= dy
                x += sx
            if e2 < dy:
                err += dx
                y += sy
                
        draw_gradient_line(canvas, lines, tint_start, tint_end)
        
        #Calculate offset for next parallel line
        if dx > dy:
            x1, y1 = x1, y1 - 1
            x2, y2 = x2, y2 - 1
        else:
            x1, y1 = x1 - 1, y1
            x2, y2 = x2 - 1, y2
        err = (dx if dx > dy else -dy) / 2 # Reset error for next line
        
def bresenhamline(start, end, max_iter=5):
    """
    Returns a list of points from (start, end] by ray tracing a line b/w the
    points.
    Parameters:
        start: An array of start points (number of points x dimension)
        end:   An end points (1 x dimension)
            or An array of end point corresponding to each start point
                (number of points x dimension)
        max_iter: Max points to traverse. if -1, maximum number of required
                  points are traversed

    Returns:
        linevox (n x dimension) A cumulative array of all points traversed by
        all the lines so far.

    >>> s = np.array([[3, 1, 9, 0],[0, 0, 3, 0]])
    >>> bresenhamline(s, np.zeros(s.shape[1]), max_iter=-1)
    array([[ 3,  1,  8,  0],
           [ 2,  1,  7,  0],
           [ 2,  1,  6,  0],
           [ 2,  1,  5,  0],
           [ 1,  0,  4,  0],
           [ 1,  0,  3,  0],
           [ 1,  0,  2,  0],
           [ 0,  0,  1,  0],
           [ 0,  0,  0,  0],
           [ 0,  0,  2,  0],
           [ 0,  0,  1,  0],
           [ 0,  0,  0,  0],
           [ 0,  0, -1,  0],
           [ 0,  0, -2,  0],
           [ 0,  0, -3,  0],
           [ 0,  0, -4,  0],
           [ 0,  0, -5,  0],
           [ 0,  0, -6,  0]])
    """
    # Return the points as a single array
    return _bresenhamlines(start, end, max_iter).reshape(-1, start.shape[-1])