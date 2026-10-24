"""Create a simple lit 3D cube mesh with no downloaded assets."""

from panda3d.core import (
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    NodePath,
)


def unit_box() -> NodePath:
    """Return a 1x1x1 cube centered at the origin, with six face normals."""
    vertex_data = GeomVertexData(
        "box", GeomVertexFormat.getV3n3(), Geom.UHStatic
    )
    vertex_data.setNumRows(24)
    vertices = GeomVertexWriter(vertex_data, "vertex")
    normals = GeomVertexWriter(vertex_data, "normal")
    triangles = GeomTriangles(Geom.UHStatic)

    a, b = -0.5, 0.5
    faces = [
        ((0, 0, 1), [(a, a, b), (b, a, b), (b, b, b), (a, b, b)]),
        ((0, 0, -1), [(a, b, a), (b, b, a), (b, a, a), (a, a, a)]),
        ((0, -1, 0), [(a, a, a), (b, a, a), (b, a, b), (a, a, b)]),
        ((0, 1, 0), [(b, b, a), (a, b, a), (a, b, b), (b, b, b)]),
        ((1, 0, 0), [(b, a, a), (b, b, a), (b, b, b), (b, a, b)]),
        ((-1, 0, 0), [(a, b, a), (a, a, a), (a, a, b), (a, b, b)]),
    ]

    for face_index, (normal, corners) in enumerate(faces):
        start = face_index * 4
        for xyz in corners:
            vertices.addData3(*xyz)
            normals.addData3(*normal)
        triangles.addVertices(start, start + 1, start + 2)
        triangles.addVertices(start, start + 2, start + 3)

    geometry = Geom(vertex_data)
    geometry.addPrimitive(triangles)
    node = GeomNode("unit-box")
    node.addGeom(geometry)
    return NodePath(node)


class _ShapeMesh:
    def __init__(self, name: str) -> None:
        self._data = GeomVertexData(name, GeomVertexFormat.getV3n3(), Geom.UHStatic)
        self._vertices = GeomVertexWriter(self._data, "vertex")
        self._normals = GeomVertexWriter(self._data, "normal")
        self._triangles = GeomTriangles(Geom.UHStatic)
        self._count = 0
        self._name = name

    def face(self, corners: list[tuple[float, float, float]]) -> None:
        from math import sqrt

        a, b, c = corners[:3]
        u = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        v = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
        n = (
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        )
        length = sqrt(sum(x * x for x in n))
        if length < 1e-10:
            return
        normal = tuple(x / length for x in n)
        first = self._count
        for position in corners:
            self._vertices.addData3(*position)
            self._normals.addData3(*normal)
            self._count += 1
        for i in range(1, len(corners) - 1):
            self._triangles.addVertices(first, first + i, first + i + 1)

    def node(self) -> NodePath:
        geom = Geom(self._data)
        geom.addPrimitive(self._triangles)
        node = GeomNode(self._name)
        node.addGeom(geom)
        return NodePath(node)


def _cylinder(radius: float, bottom: float, top: float, name: str) -> NodePath:
    from math import cos, sin, tau

    mesh = _ShapeMesh(name)
    segments = 36
    for i in range(segments):
        angle0 = tau * i / segments
        angle1 = tau * (i + 1) / segments
        x0, y0 = radius * cos(angle0), radius * sin(angle0)
        x1, y1 = radius * cos(angle1), radius * sin(angle1)
        mesh.face([(0, 0, top), (x0, y0, top), (x1, y1, top)])
        mesh.face([(0, 0, bottom), (x1, y1, bottom), (x0, y0, bottom)])
        mesh.face([
            (x0, y0, bottom), (x1, y1, bottom),
            (x1, y1, top), (x0, y0, top),
        ])
    return mesh.node()


def _raised_curve(
    points: list[tuple[float, float]], width: float, bottom: float,
    top: float, name: str,
) -> NodePath:
    from math import hypot

    mesh = _ShapeMesh(name)
    left = []
    right = []
    half = width / 2.0
    for i, (x, y) in enumerate(points):
        before = points[max(i - 1, 0)]
        after = points[min(i + 1, len(points) - 1)]
        dx, dy = after[0] - before[0], after[1] - before[1]
        length = hypot(dx, dy)
        nx, ny = -dy / length, dx / length
        left.append((x + nx * half, y + ny * half))
        right.append((x - nx * half, y - ny * half))

    for i in range(len(points) - 1):
        li, lj = left[i], left[i + 1]
        ri, rj = right[i], right[i + 1]
        mesh.face([
            (*li, top), (*ri, top), (*rj, top), (*lj, top),
        ])
        mesh.face([
            (*ri, bottom), (*li, bottom), (*lj, bottom), (*rj, bottom),
        ])
        mesh.face([
            (*li, bottom), (*li, top), (*lj, top), (*lj, bottom),
        ])
        mesh.face([
            (*ri, bottom), (*rj, bottom), (*rj, top), (*ri, top),
        ])
    mesh.face([
        (*right[0], bottom), (*right[0], top),
        (*left[0], top), (*left[0], bottom),
    ])
    mesh.face([
        (*left[-1], bottom), (*left[-1], top),
        (*right[-1], top), (*right[-1], bottom),
    ])
    return mesh.node()


def _add_box(
    parent: NodePath, box: NodePath, name: str,
    position: tuple[float, float, float],
    size: tuple[float, float, float],
    color: tuple[float, float, float, float], heading: float = 0.0,
) -> None:
    part = box.copyTo(parent)
    part.setName(name)
    part.setPos(*position)
    part.setScale(*size)
    part.setH(heading)
    part.setColor(*color)


def _add_piece(parent: NodePath, piece: NodePath, color) -> None:
    piece.reparentTo(parent)
    piece.setColor(*color)


def switch_marker(kind: str, box: NodePath) -> NodePath:
    root = NodePath(f"switch-{kind}")
    stone = (0.35, 0.36, 0.37, 1.0)

    if kind == "o":
        disc = _cylinder(0.255, 0.012, 0.075, "soft-disc")
        _add_piece(root, disc, stone)

    elif kind == "X":
        for angle in (45.0, -45.0):
            _add_box(root, box, "heavy-x", (0, 0, 0.052),
                     (0.64, 0.105, 0.08), stone, angle)

    elif kind == "P":
        for side in (-1, 1):
            points = []
            for i in range(25):
                t = -1.0 + i / 12.0
                x = side * (0.125 + 0.165 * (1.0 - t * t) ** 0.8)
                points.append((x, 0.27 * t))
            curve = _raised_curve(points, 0.085, 0.012, 0.075,
                                  "split-parenthesis")
            _add_piece(root, curve, stone)
    else:
        raise ValueError(f"Unknown switch tile: {kind}")

    return root


def bridge_model(box: NodePath) -> NodePath:
    root = NodePath("bridge-floor")
    _add_box(root, box, "bridge-tile", (0, 0, -0.09),
             (0.97, 0.97, 0.18), (0.57, 0.59, 0.61, 1.0))
    return root
