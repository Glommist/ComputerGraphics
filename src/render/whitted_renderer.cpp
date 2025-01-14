#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <vector>
#include <optional>
#include <iostream>
#include <chrono>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "render_engine.h"
#include "../scene/light.h"
#include "../utils/math.hpp"
#include "../utils/ray.h"
#include "../utils/logger.h"

using std::chrono::steady_clock;
using duration   = std::chrono::duration<float>;
using time_point = std::chrono::time_point<steady_clock, duration>;
using Eigen::Vector3f;

// 最大的反射次数
constexpr int MAX_DEPTH        = 5;
constexpr float INFINITY_FLOAT = std::numeric_limits<float>::max();
// 考虑物体与光线相交点的偏移值
constexpr float EPSILON = 0.00001f;

// 当前物体的材质类型，根据不同材质类型光线会有不同的反射情况
enum class MaterialType
{
    DIFFUSE_AND_GLOSSY,
    REFLECTION
};

// 显示渲染的进度条
void update_progress(float progress)
{
    int barwidth = 70;
    std::cout << "[";
    int pos = static_cast<int>(barwidth * progress);
    for (int i = 0; i < barwidth; i++) {
        if (i < pos)
            std::cout << "=";
        else if (i == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "]" << int(progress * 100.0) << " %\r";
    std::cout.flush();
}

WhittedRenderer::WhittedRenderer(RenderEngine& engine)
    : width(engine.width), height(engine.height), n_threads(engine.n_threads), use_bvh(false),
      rendering_res(engine.rendering_res)
{
    logger = get_logger("Whitted Renderer");
}

// whitted-style渲染的实现
void WhittedRenderer::render(Scene& scene)
{
    time_point begin_time = steady_clock::now();
    width                 = std::floor(width);
    height                = std::floor(height);

    // initialize frame buffer
    std::vector<Vector3f> framebuffer(static_cast<size_t>(width * height));
    for (auto& v : framebuffer) {
        v = Vector3f(0.0f, 0.0f, 0.0f);
    }

    int idx = 0;
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            // generate ray
            Ray ray = generate_ray(static_cast<int>(width), static_cast<int>(height), i, j,
                                   scene.camera, 1.0f);
            // cast ray
            framebuffer[idx++] = cast_ray(ray, scene, 0);
        }
        update_progress(j / height);
    }
    static unsigned char color_res[3];
    rendering_res.clear();
    for (long unsigned int i = 0; i < framebuffer.size(); i++) {
        color_res[0] = static_cast<unsigned char>(255 * clamp(0.f, 1.f, framebuffer[i][0]));
        color_res[1] = static_cast<unsigned char>(255 * clamp(0.f, 1.f, framebuffer[i][1]));
        color_res[2] = static_cast<unsigned char>(255 * clamp(0.f, 1.f, framebuffer[i][2]));
        rendering_res.push_back(color_res[0]);
        rendering_res.push_back(color_res[1]);
        rendering_res.push_back(color_res[2]);
    }
    time_point end_time         = steady_clock::now();
    duration rendering_duration = end_time - begin_time;
    logger->info("rendering takes {:.6f} seconds", rendering_duration.count());
}

// 菲涅尔定理计算反射光线
float WhittedRenderer::fresnel(const Vector3f& I, const Vector3f& N, const float& ior)
{
    float cos_theta = -N.normalized().dot(I.normalized());
    float R = ior + (1.0f - ior) * pow(1.0f - cos_theta, 5.0f);
    return R;
}

// 如果相交返回Intersection结构体，如果不相交则返回false
std::optional<std::tuple<Intersection, GL::Material>> WhittedRenderer::trace(const Ray& ray,
                                                                             const Scene& scene)
{
    std::optional<Intersection> payload;
    Eigen::Matrix4f M;
    GL::Material material;
    for (const auto& group : scene.groups) {
        for (const auto& object : group->objects) 
        {
            std::optional<Intersection> temp_payload;
            temp_payload = naive_intersect(ray, object->mesh, object->model());
            if ((!payload.has_value() && temp_payload.has_value()) ||
                (payload.has_value() && temp_payload.has_value() && temp_payload->t < payload->t))
            {
                payload  = temp_payload;
                material = object->mesh.material;
            }
            // if use bvh(exercise 2.4): use object->bvh->intersect
            // else(exercise 2.3): use naive_intersect()
            // pay attention to the range of payload->t
        }
    }
    if (!payload.has_value())
    {
        return std::nullopt;
    }
    return std::make_tuple(payload.value(), material);
}

Vector3f WhittedRenderer::cast_ray(const Ray& ray, const Scene& scene, int depth)
{
    if (depth > MAX_DEPTH) 
    {
        return Vector3f(0.0f, 0.0f, 0.0f);
    }
    Vector3f zero(0.0f, 0.0f, 0.0f);
    Vector3f hitcolor = zero;
    auto result       = trace(ray, scene); // 找到相交点
    if (result.has_value())
    {
        auto [intersection, material] = result.value();
        Vector3f normal               = intersection.normal;
        Vector3f hit_point            = ray.origin + ray.direction * intersection.t;

        Vector3f ka     = material.ambient;
        Vector3f kd     = material.diffuse;
        Vector3f ks     = material.specular;
        float shininess = material.shininess;

        // 如果 shininess >= 1000, 计算反射光
        if (shininess >= 1000.0f)
        {
            float kr = fresnel(ray.direction, normal, 1.0f);
            if (kr > 0.0f) 
            {
                Vector3f reflection_direction = ray.direction - 2 * (ray.direction.dot(normal)) * normal;
                Ray reflected_ray{hit_point, reflection_direction};
                hitcolor += kr * cast_ray(reflected_ray, scene, depth + 1);
            }
        } 
        else 
        {
            Vector3f ambientLightIntensity = {1.0f, 1.0f, 1.0f};
            for (const auto& light : scene.lights)
            {
                // Light Direction
                Vector3f light_direction = (light.position - hit_point).normalized();

                Ray shadow_ray{ hit_point, light_direction}; // shadow_ray 是从hit_point指向light.position的单位向量。
                auto shadow_result = trace(shadow_ray, scene); // 如果有返回值，则说明该点被遮挡，处于阴影中，反之则没有

                if (shadow_result.has_value())
                {
                    // View Direction
                    Vector3f view_direction = (ray.origin - hit_point).normalized();
                    // Half Vector
                    Vector3f half_direction = (light_direction + view_direction).normalized();
                    // Light Attenuation
                    float distance  = (light.position - hit_point).norm();
                    float distance2 = distance * distance;
                    // ambient
                    Vector3f ambient = ambientLightIntensity.cwiseProduct(ka);
                    ambient          = ambient / distance2;
                    hitcolor += ambient;
                    // Diffuse
                    float diff       = std::max(normal.dot(light_direction), 0.0f);
                    Vector3f diffuse = diff * light.intensity * kd;
                    diffuse          = diffuse / distance2;
                    hitcolor += diffuse;
                    // Specular
                    float spec = std::pow(std::max(0.0f, normal.dot(half_direction)), shininess);
                    Vector3f specular = spec * light.intensity * ks;
                    specular          = specular / distance2;
                    hitcolor += specular;
                }
            }
        }
    } 
    else
    {
        hitcolor = RenderEngine::background_color;
        return hitcolor;
    }
    // 确保颜色值不超过 255
    if (hitcolor.x() > 255) {hitcolor.x() = 255.0f;}
    if (hitcolor.y() > 255) {hitcolor.y() = 255.0f;}
    if (hitcolor.z() > 255) {hitcolor.z() = 255.0f;}
    return hitcolor;
}