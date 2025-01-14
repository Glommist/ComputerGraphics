#include <array>
#include <limits>
#include <tuple>
#include <vector>
#include <algorithm>
#include <cmath>
#include <mutex>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <spdlog/spdlog.h>

#include "rasterizer.h"
#include "triangle.h"
#include "../utils/math.hpp"

#include <iostream>
using Eigen::Matrix4f;
using Eigen::Vector2f;
using Eigen::Vector2i;
using Eigen::Vector3f;
using Eigen::Vector4f;
using std::fill;
using std::tuple;

void Rasterizer::worker_thread()
{
    while (true) {
        VertexShaderPayload payload;
        Triangle triangle;
        {
            if (Context::vertex_finish && Context::vertex_shader_output_queue.empty()) {
                Context::rasterizer_finish = true;
                return;
            }
            if (Context::vertex_shader_output_queue.size() < 3) {
                continue;
            }
            std::unique_lock<std::mutex> lock(Context::vertex_queue_mutex);
            if (Context::vertex_shader_output_queue.size() < 3) {
                continue;
            }
            for (size_t vertex_count = 0; vertex_count < 3; vertex_count++) {
                payload = Context::vertex_shader_output_queue.front();
                Context::vertex_shader_output_queue.pop();
                if (vertex_count == 0) {
                    triangle.world_pos[0]    = payload.world_position;
                    triangle.viewport_pos[0] = payload.viewport_position;
                    triangle.normal[0]       = payload.normal;
                } else if (vertex_count == 1) {
                    triangle.world_pos[1]    = payload.world_position;
                    triangle.viewport_pos[1] = payload.viewport_position;
                    triangle.normal[1]       = payload.normal;
                } else {
                    triangle.world_pos[2]    = payload.world_position;
                    triangle.viewport_pos[2] = payload.viewport_position;
                    triangle.normal[2]       = payload.normal;
                }
            }
        }
        rasterize_triangle(triangle);
    }
}
// 给定坐标(x,y)以及三角形的三个顶点坐标，判断(x,y)是否在三角形的内部
bool Rasterizer::inside_triangle(int x, int y, const Vector4f* vertices)
{
    Vector2f p(x, y);

    Vector2f a(vertices[0].x(), vertices[0].y());
    Vector2f b(vertices[1].x(), vertices[1].y());
    Vector2f c(vertices[2].x(), vertices[2].y());

    Vector2f ab = b - a;
    Vector2f bc = c - b;
    Vector2f ca = a - c;

    Vector2f ap = p - a;
    Vector2f bp = p - b;
    Vector2f cp = p - c;

    float cross1 = ab.x() * ap.y() - ab.y() * ap.x();
    float cross2 = bc.x() * bp.y() - bc.y() * bp.x();
    float cross3 = ca.x() * cp.y() - ca.y() * cp.x();

    return (cross1 >= 0 && cross2 >= 0 && cross3 >= 0) || (cross1 <= 0 && cross2 <= 0 && cross3 <= 0);
}
double triangleArea(Vector2f A, Vector2f B, Vector2f C)
{
    return fabs(A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1])) / 2.0;
}
// 给定坐标(x,y)以及三角形的三个顶点坐标，计算(x,y)对应的重心坐标[alpha, beta, gamma]
tuple<float, float, float> Rasterizer::compute_barycentric_2d(float x, float y, const Vector4f* v)
{
    float alpha = 0.0f, beta = 0.0f, gamma = 0.0f;
    Vector2f A(v[0].x(), v[0].y());
    Vector2f B(v[1].x(), v[1].y());
    Vector2f C(v[2].x(), v[2].y());
    Vector2f P(x, y);
    double areaABC = triangleArea(A, B, C);
    double areaPBC = triangleArea(P, B, C);
    double areaAPC = triangleArea(A, P, C);
    double areaABP = triangleArea(A, B, P);

    alpha = float(areaPBC / areaABC);
    beta  = float(areaAPC / areaABC);
    gamma = float(areaABP / areaABC);
    return {alpha, beta, gamma};
}

// 对顶点的某一属性插值
Vector3f Rasterizer::interpolate(float alpha, float beta, float gamma, const Eigen::Vector3f& vert1,
                                 const Eigen::Vector3f& vert2, const Eigen::Vector3f& vert3,
                                 const Eigen::Vector3f& weight, const float& Z)
{
    Vector3f interpolated_res;
    for (int i = 0; i < 3; i++) {
        interpolated_res[i] = alpha * vert1[i] / weight[0] + beta * vert2[i] / weight[1] +
                              gamma * vert3[i] / weight[2];
    }
    interpolated_res *= Z;
    return interpolated_res;
    //相当于做了一个“加权平均”
}
float Mymax(float a, float b, float c)
{
    if (a > b)
        if (a > c)
            return a;
        else
            return c;
    else if (b > c)
        return b;
    return c;
}
float Mymin(float a, float b, float c)
{
    if (a < b)
        if (a < c)
            return a;
        else
            return c;
    else if (b < c)
        return b;
    return c;
}
// 对当前三角形进行光栅化
void Rasterizer::rasterize_triangle(Triangle& t)
{
    Vector4f v[3];
    v[0] = t.viewport_pos[0];
    v[1] = t.viewport_pos[1];
    v[2] = t.viewport_pos[2];

    int x_min = std::max(0, static_cast<int>(std::floor(Mymin(v[0].x(), v[1].x(), v[2].x()))));
    int x_max = std::min(Context::frame_buffer.width - 1,
                         static_cast<int>(std::ceil(Mymax(v[0].x(), v[1].x(), v[2].x()))));
    int y_min = std::max(0, static_cast<int>(std::floor(Mymin(v[0].y(), v[1].y(), v[2].y()))));
    int y_max = std::min(Context::frame_buffer.height - 1,
                         static_cast<int>(std::ceil(Mymax(v[0].y(), v[1].y(), v[2].y()))));
    /*计算三角框边界，只需要找出给定的三角形的三个点中的xy的最大最小值即可，
    且在取最小值和最大值时间分别采用了floor和ceil，以确保取到三角形内的所有值
    同时，为了不溢出缓冲区，分别再与0和Context::frame_buffer.width - 1比较。
    得到了矩形边界框*/
    for (int x = x_min; x <= x_max; ++x) {
        for (int y = y_min; y <= y_max; ++y) {
            if (inside_triangle(x, y, v)) {
                auto [alpha, beta, gamma] = compute_barycentric_2d((float)x, (float)y, v);
                float w_reciprocal1       = 1.0f / v[0].w();
                float w_reciprocal2       = 1.0f / v[1].w();
                float w_reciprocal3       = 1.0f / v[2].w();
                Vector3f temp;
                temp[0] = w_reciprocal1;
                temp[1] = w_reciprocal2;
                temp[2] = w_reciprocal3;
                float Z =
                    1.0f / (alpha * w_reciprocal1 + beta * w_reciprocal2 + gamma * w_reciprocal3);
                float depth = alpha * v[0].z() * w_reciprocal1 + beta * v[1].z() * w_reciprocal2 +
                              gamma * v[2].z() * w_reciprocal3;
                depth *= Z;
                Vector3f interpolated_normal =
                    interpolate(alpha, beta, gamma, t.normal[0], t.normal[1], t.normal[2], temp, Z);
                Vector3f interpolated_position =
                    interpolate(alpha, beta, gamma, t.world_pos[0].head<3>(),
                                t.world_pos[1].head<3>(), t.world_pos[2].head<3>(), temp, Z);
                FragmentShaderPayload payload;
                payload.x            = x;
                payload.y            = y;
                payload.world_pos    = interpolated_position;
                payload.world_normal = interpolated_normal;
                payload.depth        = depth;
                {
                    std::unique_lock<std::mutex> lock(Context::rasterizer_queue_mutex);
                    Context::rasterizer_output_queue.push(payload);
                }
            }
        }
    }
}
