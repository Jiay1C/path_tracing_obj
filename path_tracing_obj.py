import taichi as ti
from vispy import io

ti.init(arch=ti.cuda)

# Args
WIDTH = 1000
HEIGHT = 1000
VIEWPORT_RATIO = 0.001
VIEWPORT_WIDTH = WIDTH * VIEWPORT_RATIO
VIEWPORT_HEIGHT = HEIGHT * VIEWPORT_RATIO
VIEWPORT_FOCAL_LENGTH = 1
SAMPLES_PER_PIXEL = 32
MAX_DEPTH = 10
PI = 3.1415926535

# Global Variables
FrameBuffer = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
CAMERA_ORIGIN = ti.Vector.field(4, dtype=ti.f32, shape=())
CAMERA_ORIGIN[None] = [0, 0, VIEWPORT_FOCAL_LENGTH, 1]


@ti.func
def vector4d_cross(a, b):
    a_3d = ti.Vector([a[0], a[1], a[2]])
    b_3d = ti.Vector([b[0], b[1], b[2]])
    res_3d = a_3d.cross(b_3d)
    return ti.Vector([res_3d[0], res_3d[1], res_3d[2], 0])


@ti.func
def random_vector():
    p = 2.0 * ti.Vector([ti.random(), ti.random(), ti.random(), 0]) - ti.Vector([1, 1, 1, 0])
    while p.norm() >= 1.0:
        p = 2.0 * ti.Vector([ti.random(), ti.random(), ti.random(), 0]) - ti.Vector([1, 1, 1, 0])
    return p


@ti.data_oriented
class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    def at(self, t):
        return self.origin + self.direction * t


@ti.data_oriented
class Triangle:
    def __init__(self, vertex_a, vertex_b, vertex_c):
        self.vertex_a = vertex_a
        self.vertex_b = vertex_b
        self.vertex_c = vertex_c

    @ti.func
    def hit(self, ray, t_min=1e-3, t_max=1e9, s_diff_min=1e-4):
        is_hit = False
        front_face = False
        root = 0.0
        hit_point = ti.Vector([0.0, 0.0, 0.0, 0.0])
        hit_point_normal = ti.Vector([0.0, 0.0, 0.0, 0.0])
        normal = vector4d_cross((self.vertex_b - self.vertex_a), (self.vertex_c - self.vertex_a))
        if ray.direction.dot(normal) != 0:
            oc = ray.origin - self.vertex_a
            root = -oc.dot(normal) / ray.direction.dot(normal)
            if root >= t_min and root <= t_max:
                hit_point = ray.at(root)
                line_a = self.vertex_a - hit_point
                line_b = self.vertex_b - hit_point
                line_c = self.vertex_c - hit_point
                s_calc = (vector4d_cross(line_a, line_b).norm() +
                          vector4d_cross(line_b, line_c).norm() +
                          vector4d_cross(line_c, line_a).norm())
                s_real = normal.norm()
                if s_calc <= s_real + s_diff_min:
                    is_hit = True
                    hit_point_normal = normal.normalized()
                    if ray.direction.dot(hit_point_normal) < 0:
                        front_face = True
                    else:
                        hit_point_normal = -hit_point_normal
        return is_hit, root, hit_point, hit_point_normal, front_face


@ti.data_oriented
class Model:
    def __init__(self, max_vertex_num=1024, max_faces_num=1024, color=ti.Vector([0.0, 0.0, 0.0]), material=1):
        self.vertex_num = ti.field(dtype=ti.i32, shape=())
        self.vertex_num[None] = 0
        self.face_num = ti.field(dtype=ti.i32, shape=())
        self.face_num[None] = 0
        self.vertex = ti.Vector.field(4, dtype=ti.f32, shape=max_vertex_num)
        self.face = ti.Vector.field(3, dtype=ti.i32, shape=max_faces_num)
        self.color = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.color[None] = color
        self.material = ti.field(dtype=ti.i32, shape=())
        self.material[None] = material

    def set_color(self, color):
        self.color[None] = color

    def set_material(self, material):
        self.material[None] = material

    def add_face(self, vertex):
        for index in range(3):
            self.vertex[self.vertex_num[None] + index] = ti.Vector(
                [vertex[index, 0], vertex[index, 1], vertex[index, 2], 1])
        self.face[self.face_num[None]] = ti.Vector(
            [self.vertex_num[None], self.vertex_num[None] + 1, self.vertex_num[None] + 2])
        self.vertex_num[None] += 3
        self.face_num[None] += 1

    def clear_face(self):
        self.vertex_num[None] = 0
        self.face_num[None] = 0

    def from_obj(self, filename):
        self.clear_face()
        vertex, face, normal, texture = io.read_mesh(filename)
        for index in range(len(vertex)):
            self.vertex[index] = ti.Vector([vertex[index][0], vertex[index][1], vertex[index][2], 1])
        for index in range(len(face)):
            self.face[index] = ti.Vector([face[index][0], face[index][1], face[index][2]])
        self.vertex_num[None] = len(vertex)
        self.face_num[None] = len(face)

    def transform(self, matrix):
        for index in range(self.vertex_num[None]):
            self.vertex[index] = matrix @ self.vertex[index]

    @ti.func
    def hit(self, ray, t_min=1e-3, t_max=1e9):
        t_closest = t_max
        is_hit = False
        front_face = False
        hit_point = ti.Vector([0.0, 0.0, 0.0, 0.0])
        hit_point_normal = ti.Vector([0.0, 0.0, 0.0, 0.0])
        for index in range(self.face_num[None]):
            triangle = Triangle(self.vertex[self.face[index][0]],
                                self.vertex[self.face[index][1]],
                                self.vertex[self.face[index][2]])
            is_hit_tmp, root_tmp, hit_point_tmp, hit_point_normal_tmp, front_face_tmp = \
                triangle.hit(ray, t_min, t_closest)
            if is_hit_tmp:
                t_closest = root_tmp
                is_hit = is_hit_tmp
                hit_point = hit_point_tmp
                hit_point_normal = hit_point_normal_tmp
                front_face = front_face_tmp
        return is_hit, t_closest, hit_point, hit_point_normal, front_face, self.material[None], self.color[None]


@ti.data_oriented
class ModelSphere:
    def __init__(self, center, radius, material, color):
        self.center = center
        self.radius = radius
        self.material = material
        self.color = color

    @ti.func
    def hit(self, ray, t_min=1e-3, t_max=1e9):
        is_hit = False
        front_face = False
        root = 0.0
        hit_point = ti.Vector([0.0, 0.0, 0.0, 0.0])
        hit_point_normal = ti.Vector([0.0, 0.0, 0.0, 0.0])
        oc = ray.origin - self.center
        a = ray.direction.dot(ray.direction)
        hb = oc.dot(ray.direction)
        c = oc.dot(oc) - self.radius * self.radius
        discriminant = hb * hb - a * c
        if discriminant > 0:
            sqrt_discriminant = ti.sqrt(discriminant)
            root = (-hb - sqrt_discriminant) / a
            if root < t_min or root > t_max:
                root = (-hb + sqrt_discriminant) / a
                if root >= t_min and root <= t_max:
                    is_hit = True
            else:
                is_hit = True
        if is_hit:
            hit_point = ray.at(root)
            hit_point_normal = (hit_point - self.center) / self.radius
            if ray.direction.dot(hit_point_normal) < 0:
                front_face = True
            else:
                hit_point_normal = -hit_point_normal
        return is_hit, root, hit_point, hit_point_normal, front_face, self.material, self.color


@ti.data_oriented
class Scene:
    def __init__(self):
        self.model = []

    def add_model(self, model):
        self.model.append(model)

    def clear_model(self):
        self.model.clear()

    @ti.func
    def hit(self, ray, t_min=1e-3, t_max=1e9):
        t_closest = t_max
        is_hit = False
        front_face = False
        hit_point = ti.Vector([0.0, 0.0, 0.0, 0.0])
        hit_point_normal = ti.Vector([0.0, 0.0, 0.0, 0.0])
        color = ti.Vector([0.0, 0.0, 0.0])
        material = 1
        for index in ti.static(range(len(self.model))):
            is_hit_tmp, root_tmp, hit_point_tmp, hit_point_normal_tmp, front_face_tmp, material_tmp, color_tmp = \
                self.model[index].hit(ray, t_min, t_closest)
            if is_hit_tmp:
                t_closest = root_tmp
                is_hit = is_hit_tmp
                hit_point = hit_point_tmp
                hit_point_normal = hit_point_normal_tmp
                front_face = front_face_tmp
                material = material_tmp
                color = color_tmp
        return is_hit, hit_point, hit_point_normal, front_face, material, color


Scene = Scene()


@ti.func
def reflect(v, normal):
    return v - 2 * v.dot(normal) * normal


@ti.func
def refract(uv, n, etai_over_etat):
    cos_theta = min(n.dot(-uv), 1.0)
    r_out_perp = etai_over_etat * (uv + cos_theta * n)
    r_out_parallel = -ti.sqrt(abs(1.0 - r_out_perp.dot(r_out_perp))) * n
    return r_out_perp + r_out_parallel


@ti.func
def reflectance(cosine, ref_idx):
    # Use Schlick's approximation for reflectance.
    r0 = (1 - ref_idx) / (1 + ref_idx)
    r0 = r0 * r0
    return r0 + (1 - r0) * pow((1 - cosine), 5)


@ti.func
def ray_tracing(ray):
    ray_color = ti.Vector([0.0, 0.0, 0.0])
    ray_color_loss = ti.Vector([1.0, 1.0, 1.0])
    ray_origin = ray.origin
    ray_direction = ray.direction
    for depth in range(MAX_DEPTH):
        is_hit, hit_point, hit_point_normal, front_face, material, color = Scene.hit(Ray(ray_origin, ray_direction),
                                                                                     1e-5, 1e9)
        if is_hit:
            # Light
            if material == 0:
                ray_color = color * ray_color_loss
                break
            # Diffuse
            elif material == 1:
                ray_color_loss *= color
                ray_origin = hit_point
                ray_direction = hit_point_normal.normalized() + random_vector().normalized()
            elif material == 2:
                ray_color_loss *= color
                refraction_ratio = 1.5
                if front_face:
                    refraction_ratio = 1 / refraction_ratio
                ray_direction = ray_direction.normalized()
                cos_theta = min(-ray_direction.dot(hit_point_normal), 1.0)
                sin_theta = ti.sqrt(1 - cos_theta * cos_theta)
                if refraction_ratio * sin_theta > 1.0 or reflectance(cos_theta, refraction_ratio) > ti.random():
                    ray_direction = reflect(ray_direction, hit_point_normal)
                else:
                    ray_direction = refract(ray_direction, hit_point_normal, refraction_ratio)
                ray_origin = hit_point
            elif material == 3:
                ray_color_loss *= color
                ray_origin = hit_point
                ray_direction = reflect(ray_direction.normalized(), hit_point_normal)
            elif material == 4:
                ray_color_loss *= color
                fuzz = 0.4
                ray_origin = hit_point
                ray_direction = reflect(ray_direction.normalized() + fuzz * random_vector().normalized(),
                                        hit_point_normal)
                if ray_direction.dot(hit_point_normal) < 0:
                    break
        else:
            break
    return ray_color


@ti.kernel
def render():
    for i, j in FrameBuffer:
        u = (i + ti.random()) * VIEWPORT_RATIO - VIEWPORT_WIDTH / 2
        v = (j + ti.random()) * VIEWPORT_RATIO - VIEWPORT_HEIGHT / 2
        pixel_position = ti.Vector([u, v, 0.0, 1.0])
        color = ti.Vector([0.0, 0.0, 0.0])
        for n in range(SAMPLES_PER_PIXEL):
            ray = Ray(origin=CAMERA_ORIGIN[None], direction=pixel_position - CAMERA_ORIGIN[None])
            color += ray_tracing(ray)
        color /= SAMPLES_PER_PIXEL
        FrameBuffer[i, j] = color


def scene_init_box():
    light = Model(material=0, color=ti.Vector([5, 5, 5]))
    light.from_obj("model/light_square.obj")
    box = Model(material=1, color=ti.Vector([1.0, 1.0, 1.0]))
    box.from_obj("model/box_wall.obj")
    red_wall = Model(material=1, color=ti.Vector([1.0, 0.0, 0.0]))
    red_wall.from_obj("model/left_red.obj")
    green_wall = Model(material=1, color=ti.Vector([0.0, 1.0, 0.0]))
    green_wall.from_obj("model/right_green.obj")
    Scene.add_model(light)
    Scene.add_model(box)
    Scene.add_model(red_wall)
    Scene.add_model(green_wall)


def scene_init_block():
    block = Model(material=1, color=ti.Vector([0.9, 0.9, 0.9]), max_vertex_num=16384, max_faces_num=16384)
    block.from_obj("model/block.obj")
    T1 = ti.Matrix([[ti.cos(PI * 0.3), 0, -ti.sin(PI * 0.3), 0],
                    [0, 1, 0, 0],
                    [ti.sin(PI * 0.3), 0, ti.cos(PI * 0.3), 0],
                    [0, 0, 0, 1]])
    T2 = ti.Matrix([[0.04, 0, 0, 0],
                    [0, 0.04, 0, -0.6],
                    [0, 0, 0.04, -2],
                    [0, 0, 0, 1]])
    T = T2 @ T1
    block.transform(T)
    Scene.add_model(block)


def scene_init_dinosaur():
    dinosaur = Model(material=4, color=ti.Vector([0.7, 0.7, 0.7]), max_vertex_num=8192, max_faces_num=8192)
    dinosaur.from_obj("model/dinosaur.obj")
    T1 = ti.Matrix([[1, 0, 0, 0],
                    [0, ti.cos(PI * 0.5), ti.sin(PI * 0.5), 0],
                    [0, -ti.sin(PI * 0.5), ti.cos(PI * 0.5), 0],
                    [0, 0, 0, 1]])
    T2 = ti.Matrix([[ti.cos(PI * 0.2), 0, -ti.sin(PI * 0.2), 0],
                    [0, 1, 0, 0],
                    [ti.sin(PI * 0.2), 0, ti.cos(PI * 0.2), 0],
                    [0, 0, 0, 1]])
    T3 = ti.Matrix([[0.02, 0, 0, 0],
                    [0, 0.02, 0, -0.5],
                    [0, 0, 0.02, -3],
                    [0, 0, 0, 1]])
    T = T3 @ T2 @ T1
    dinosaur.transform(T)
    Scene.add_model(dinosaur)


def scene_init_horse():
    horse = Model(material=1, color=ti.Vector([0.9, 0.9, 0.9]), max_vertex_num=100000, max_faces_num=300000)
    horse.from_obj("model/horse.obj")
    T1 = ti.Matrix([[1, 0, 0, 0],
                    [0, ti.cos(PI * 0.5), ti.sin(PI * 0.5), 0],
                    [0, -ti.sin(PI * 0.5), ti.cos(PI * 0.5), 0],
                    [0, 0, 0, 1]])
    T2 = ti.Matrix([[ti.cos(PI * 0.8), 0, -ti.sin(PI * 0.8), 0],
                    [0, 1, 0, 0],
                    [ti.sin(PI * 0.8), 0, ti.cos(PI * 0.8), 0],
                    [0, 0, 0, 1]])
    T3 = ti.Matrix([[0.6, 0, 0, 0],
                    [0, 0.6, 0, -0.8],
                    [0, 0, 0.6, -2.3],
                    [0, 0, 0, 1]])
    T = T3 @ T2 @ T1
    horse.transform(T)
    Scene.add_model(horse)


def scene_init_round():
    round = Model(material=2, color=ti.Vector([0.9, 0.9, 0.9]), max_vertex_num=8192, max_faces_num=8192)
    round.from_obj("model/round.obj")
    round.transform(ti.Matrix([[0.003, 0, 0, 0],
                               [0, 0.003, 0, -0.5],
                               [0, 0, 0.003, -2.2],
                               [0, 0, 0, 1]]))
    Scene.add_model(round)


if __name__ == '__main__':
    gui = ti.GUI("Path Tracer", res=(WIDTH, HEIGHT))
    FrameBuffer.fill(0)
    scene_init_box()
    scene_init_dinosaur()
    # scene_init_round()
    # scene_init_block()
    # scene_init_horse()
    render()
    while gui.running:
        gui.set_image(FrameBuffer)
        gui.show()
