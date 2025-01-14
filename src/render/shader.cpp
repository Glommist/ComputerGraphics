#include "rasterizer_renderer.h"
#include "../utils/math.hpp"
#include <cstdio>


#ifdef _WIN32
#undef min
#undef max
#endif
using Eigen::Matrix4f;
using Eigen::Vector3f;
using Eigen::Vector4f;

// vertex shader
VertexShaderPayload vertex_shader(const VertexShaderPayload& payload)
{
    VertexShaderPayload output_payload = payload;
    // Vertex position transformation
    Vector4f clip_position         = Uniforms::MVP * output_payload.world_position;//这里的world_position其实是model_position。
    Vector4f ndc_position          = clip_position / clip_position.w();
    Matrix4f world_matrix_inv      = Uniforms::inv_trans_M.transpose();
    Matrix4f world_matrix          = world_matrix_inv.inverse();
    Vector4f world_position        = world_matrix * payload.world_position;
    output_payload.world_position  = world_position;
    // Viewport transformation
    Matrix4f viewport_matrix = Matrix4f::Identity();
    viewport_matrix(0, 0)    = Uniforms::width / 2.0f;
    viewport_matrix(1, 1)    = Uniforms::height / 2.0f;
    viewport_matrix(2, 2)    = (Uniforms::camera.far_plane - Uniforms::camera.near_plane) / 2.0f;
    viewport_matrix(3, 3)    = 1.0f;
    viewport_matrix(0, 3)    = Uniforms::width / 2.0f;
    viewport_matrix(1, 3)    = Uniforms::height / 2.0f;
    viewport_matrix(2, 3)    = (Uniforms::camera.far_plane + Uniforms::camera.near_plane) / 2.0f;
    Vector4f viewport_position       = viewport_matrix * ndc_position;
    output_payload.viewport_position = viewport_position;
    output_payload.normal = (Uniforms::inv_trans_M * payload.normal.homogeneous()).head<3>();
    //homogeneous()函数可以转化坐标为齐次坐标。
    //非等比例变换时，借助切线与法线始终相互垂直的性质得到法向量。
    //需要model_matrix的逆矩阵再转置得到的矩阵，乘以原本的法向量，就可以得到变换后的法向量。
    //此处的homogeneous()函数的作用为：将原本的3维normal转换到4维，用于和矩阵相乘。
    return output_payload;
}

Vector3f phong_fragment_shader(const FragmentShaderPayload& payload, const GL::Material& material,
                               const std::list<Light>& lights, const Camera& camera)
{
    Vector3f result = {0, 0, 0};

    // ka,kd,ks can be got from material.ambient,material.diffuse,material.specular
    Vector3f ka = material.ambient;
    Vector3f kd = material.diffuse;
    Vector3f ks = material.specular;

    float shininess = material.shininess;
    // set ambient light intensity
    Vector3f ambientLightIntensity = {1.0f, 1.0f, 1.0f};
    //此处环境光大小的取舍并未进行相关文献阅读或者考证。
    for (const auto& light : lights) {
        // Light Direction
        Vector3f lightDirection = (light.position - payload.world_pos).normalized();
        // View Direction
        Vector3f viewDirection = (camera.position - payload.world_pos).normalized();
        // Half Vector
        Vector3f halfDirection = (lightDirection + viewDirection).normalized();
        //// Light Attenuation
        float distance   = (light.position - payload.world_pos.head<3>()).norm();
        float distance2 = distance * distance;
        //Ambient
        Vector3f ambient = ambientLightIntensity.cwiseProduct(ka);
        //cwiseProduct:元素乘积。
        // Vector3f ambient = light.intensity * ka;
        ambient = ambient / distance2;
        result += ambient;
        // Diffuse
        float diff       = std::max(payload.world_normal.dot(lightDirection), 0.0f);
        //点乘结果小于0,说明夹角大于90
        Vector3f diffuse = diff * light.intensity * kd;
        diffuse = diffuse /distance2;
        result += diffuse;
        // Specular
        float spec = std::pow(std::max(0.0f, payload.world_normal.dot(halfDirection)), shininess);
        Vector3f specular = spec * light.intensity * ks;
        specular   = specular / distance2;
        result += specular;
    }
    // set rendering result max threshold to 255
    result = result * 255;
    //mycount++;
    if (result.x() > 255) {result.x() = 255.0f;}
    if (result.y() > 255) {result.y() = 255.0f;}
    if (result.z() > 255) {result.z() = 255.0f;}
    //防止溢出导致的颜色显示错误。
    return result;
}
