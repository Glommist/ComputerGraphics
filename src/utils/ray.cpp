#include "ray.h"

#include <cmath>
#include <array>

#include <Eigen/Dense>
#include <spdlog/spdlog.h>

#include "../utils/math.hpp"
#include <iostream> 
using Eigen::Matrix3f;
using Eigen::Matrix4f;
using Eigen::Vector2f;
using Eigen::Vector3f;
using Eigen::Vector4f;
using std::numeric_limits;
using std::optional;
using std::size_t;

constexpr float infinity = 1e5f;
constexpr float eps      = 1e-5f;

Intersection::Intersection() : t(numeric_limits<float>::infinity()), face_index(0)
{
}

Ray generate_ray(int width, int height, int x, int y, Camera& camera, float depth)
{
    Vector2f pos((float)x + 0.5f, (float)y + 0.5f);
    Vector2f center((float)width / 2.0f, (float)height / 2.0f);
    Matrix4f inv_view = camera.view().inverse();
    Vector4f view_pos_specified  = {pos.x() - center.x(), -(pos.y() - center.y()), -depth, 1.0f};
    
    float fov_y                  = radians(camera.fov_y_degrees);
    float image_plane_height     = tan(fov_y / 2.0f);
    float half_heigth            = (float)height / 2.0f;
    float ratio                  = image_plane_height / half_heigth;
    view_pos_specified[2]        = -depth / ratio;
    Vector4f world_pos           = (inv_view * view_pos_specified);
    return {camera.position, (world_pos.head<3>() - camera.position).normalized()};
}

optional<Intersection> ray_triangle_intersect(const Ray& ray, const GL::Mesh& mesh, size_t index)
{
    // these lines below are just for compiling and can be deleted
    (void)ray;
    (void)mesh;
    (void)index;
    // these lines above are just for compiling and can be deleted
    Intersection result;
    
    if (result.t - infinity < -eps) {
        return result;
    } else {
        return std::nullopt;
    }
}

optional<Intersection> naive_intersect(const Ray& ray, const GL::Mesh& mesh, const Matrix4f model)
{
    Intersection result;
    for (size_t i = 0; i < mesh.faces.count(); ++i) {
        std::array<size_t, 3> v_indices = mesh.face(i);
        (void)v_indices;

        Vector3f x0_model = mesh.vertex(v_indices[0]);
        Vector3f x1_model = mesh.vertex(v_indices[1]);
        Vector3f x2_model = mesh.vertex(v_indices[2]);

        Vector4f x0_world;
        x0_world[0]       = x0_model[0];
        x0_world[1]       = x0_model[1];
        x0_world[2]       = x0_model[2];
        x0_world[3]       = 1.0f;
        x0_world          = model * x0_world;

        Vector4f x1_world;
        x1_world[0] = x1_model[0];
        x1_world[1] = x1_model[1];
        x1_world[2] = x1_model[2];
        x1_world[3] = 1.0f;
        x1_world    = model * x1_world;

        Vector4f x2_world;
        x2_world[0] = x2_model[0];
        x2_world[1] = x2_model[1];
        x2_world[2] = x2_model[2];
        x2_world[3] = 1.0f;
        x2_world    = model * x2_world;

        Vector3f V0    = x0_world.head<3>();
        Vector3f V1    = x1_world.head<3>();
        Vector3f V2    = x2_world.head<3>();
        Vector3f edge1 = V1 - V0;
        Vector3f edge2 = V2 - V0;

        Vector3f direction = ray.direction;
        Vector3f h         = direction.cross(edge2);
        float a            = edge1.dot(h);

        if (fabs(a) < eps)
            continue;

        float f = 1.0f / a;
        Vector3f s = ray.origin - V0;
        float u    = f * s.dot(h);
        if (u < 0.0f || u > 1.0f)
            continue;

        Vector3f q = s.cross(edge1);
        float v    = f * direction.dot(q);
        if (v < 0.0f || u + v > 1.0f)
            continue;

        float t = f * edge2.dot(q);
        if (t > eps && t < result.t)//find the lowest t.
        {
            result.t = t;
            result.face_index = i;
            result.barycentric_coord = Eigen::Vector3f(u, v, 1.0f - u - v);
            result.normal = (edge1.cross(edge2)).normalized();
        }
        // Vertex a, b and c are assumed to be in counterclockwise order.
        // Construct matrix A = [d, a - b, a - c] and solve Ax = (a - origin)
        // Matrix A is not invertible, indicating the ray is parallel with the triangle.
        // Test if alpha, beta and gamma are all between 0 and 1.
    }
    // Ensure result.t is strictly less than the constant `infinity`.
    if (result.t - infinity < -eps) {
        return result;
    }
    return std::nullopt;
}
