#include "../Matrix4.hpp"

namespace Athena
{
    Matrix4::Matrix4()
    {
        data[0] = 1, data[1] = 0, data[2] = 0, data[3] = 0,
        data[4] = 0, data[5] = 1, data[6] = 0, data[7] = 0,
        data[8] = 0, data[9] = 0, data[10] = 1, data[11] = 0,
        data[12] = 0, data[13] = 0, data[14] = 0, data[15] = 1;
    }

    Matrix4::Matrix4(const Matrix4& mat)
    {
        data[0] = mat.data[0], data[1] = mat.data[1], data[2] = mat.data[2], data[3] = mat.data[3],
        data[4] = mat.data[4], data[5] = mat.data[5], data[6] = mat.data[6], data[7] = mat.data[7],
        data[8] = mat.data[8], data[9] = mat.data[9], data[10] = mat.data[10], data[11] = mat.data[11],
        data[12] = mat.data[12], data[13] = mat.data[13], data[14] = mat.data[14], data[15] = mat.data[15];
    }

    Matrix4::Matrix4(const Scalar& v1, const Scalar& v2, const Scalar& v3, const Scalar& v4,
                    const Scalar& v5, const Scalar& v6, const Scalar& v7, const Scalar& v8,
                    const Scalar& v9, const Scalar& v10, const Scalar& v11, const Scalar& v12,
                    const Scalar& v13, const Scalar& v14, const Scalar& v15, const Scalar& v16)
    {
        data[0] = v1, data[1] = v2, data[2] = v3, data[3] = v4,
        data[4] = v5, data[5] = v6, data[6] = v7, data[7] = v8,
        data[8] = v9, data[9] = v10, data[10] = v11, data[11] = v12,
        data[12] = v13, data[13] = v14, data[14] = v15, data[15] = v16;
    }

    Matrix4::Matrix4(const Vector4& vec1, const Vector4& vec2, const Vector4& vec3, const Vector4& vec4)
    {
        data[0] = vec1[0];
        data[4] = vec1[1];
        data[8] = vec1[2];
        data[12] = vec1[3];

        data[1] = vec2[0];
        data[5] = vec2[1];
        data[9] = vec2[2];
        data[13] = vec2[3];

        data[2] = vec3[0];
        data[6] = vec3[1];
        data[9] = vec3[2];
        data[14] = vec3[3];

        data[3] = vec4[0];
        data[7] = vec4[1];
        data[10] = vec4[2];
        data[15] = vec4[3];
    }

    void Matrix4::operator*=(const Matrix4& mat)
    {
        Scalar temp1, temp2, temp3, temp4;
        temp1 = data[0] * mat.data[0] + data[1] * mat.data[4] + data[2] * mat.data[8] + data[3] * mat.data[12];
        temp2 = data[0] * mat.data[1] + data[1] * mat.data[5] + data[2] * mat.data[9] + data[3] * mat.data[13];
        temp3 = data[0] * mat.data[2] + data[1] * mat.data[6] + data[2] * mat.data[10] + data[3] * mat.data[14];
        temp4 = data[0] * mat.data[3] + data[1] * mat.data[7] + data[2] * mat.data[11] + data[3] * mat.data[15];

        data[0] = temp1;
        data[1] = temp2;
        data[2] = temp3;
        data[3] = temp4;

        temp1 = data[4] * mat.data[0] + data[5] * mat.data[4] + data[6] * mat.data[8] + data[7] * mat.data[12];
        temp2 = data[4] * mat.data[1] + data[5] * mat.data[5] + data[6] * mat.data[9] + data[7] * mat.data[13];
        temp3 = data[4] * mat.data[2] + data[5] * mat.data[6] + data[6] * mat.data[10] + data[7] * mat.data[14];
        temp4 = data[4] * mat.data[3] + data[5] * mat.data[7] + data[6] * mat.data[11] + data[7] * mat.data[15];

        data[4] = temp1;
        data[5] = temp2;
        data[6] = temp3;
        data[7] = temp4;

        temp1 = data[8] * mat.data[0] + data[9] * mat.data[4] + data[10] * mat.data[8] + data[11] * mat.data[12];
        temp2 = data[8] * mat.data[1] + data[9] * mat.data[5] + data[10] * mat.data[9] + data[11] * mat.data[13];
        temp3 = data[8] * mat.data[2] + data[9] * mat.data[6] + data[10] * mat.data[10] + data[11] * mat.data[14];
        temp4 = data[8] * mat.data[3] + data[9] * mat.data[7] + data[10] * mat.data[11] + data[11] * mat.data[15];

        data[8] = temp1;
        data[9] = temp2;
        data[10] = temp3;
        data[11] = temp4;

        temp1 = data[12] * mat.data[0] + data[13] * mat.data[4] + data[14] * mat.data[8] + data[15] * mat.data[12];
        temp2 = data[12] * mat.data[1] + data[13] * mat.data[5] + data[14] * mat.data[9] + data[15] * mat.data[13];
        temp3 = data[12] * mat.data[2] + data[13] * mat.data[6] + data[14] * mat.data[10] + data[15] * mat.data[14];
        temp4 = data[12] * mat.data[3] + data[13] * mat.data[7] + data[14] * mat.data[11] + data[15] * mat.data[15];

        data[12] = temp1;
        data[13] = temp2;
        data[14] = temp3;
        data[15] = temp4;
    }

    void Matrix4::operator*=(const Scalar& value)
    {
        for(unsigned int i = 0; i < 16; ++i)
        {
            data[i] *= value;
        }
    }

    void Matrix4::operator+=(const Matrix4& mat)
    {
        for(unsigned int i = 0; i < 16; ++i)
        {
            data[i] += mat.data[i];
        }
    }

    void Matrix4::operator-=(const Matrix4& mat)
    {
        for(unsigned int i = 0; i < 16; ++i)
        {
            data[i] -= mat.data[i];
        }
    }

    Matrix4 Matrix4::operator*(const Matrix4& mat) const
    {
        Matrix4 result = Matrix4(*this);
        result *= mat;
        return result;
    }

    Matrix4 Matrix4::operator*(const Scalar& value) const
    {
        Matrix4 result = Matrix4(*this);
        result *= value;
        return result;
    }

    Matrix4 Matrix4::operator+(const Matrix4& mat) const
    {
        Matrix4 result = Matrix4(*this);
        result += mat;
        return result;
    }

    Matrix4 Matrix4::operator-(const Matrix4& mat) const
    {
        Matrix4 result = Matrix4(*this);
        result -= mat;
        return result;
    }

    Matrix4 Matrix4::operator-() const
    {
        return Matrix4(-data[0], -data[1], -data[2], -data[3],
                       -data[4], -data[5], -data[6], -data[7],
                       -data[8], -data[9], -data[10], -data[11],
                       -data[12], -data[13], -data[14], -data[15]);
    }

    void Matrix4::operator=(const Matrix4& mat)
    {
        for(unsigned int i = 0; i < 16; ++i)
        {
            data[i] = mat.data[i];
        }
    }

    void Matrix4::setInverse(const Matrix4& mat)
    {
        double inv[16], det;
        int i;

        inv[0] = mat.data[5]  * mat.data[10] * mat.data[15] - 
                mat.data[5]  * mat.data[11] * mat.data[14] - 
                mat.data[9]  * mat.data[6]  * mat.data[15] + 
                mat.data[9]  * mat.data[7]  * mat.data[14] +
                mat.data[13] * mat.data[6]  * mat.data[11] - 
                mat.data[13] * mat.data[7]  * mat.data[10];

        inv[4] = -mat.data[4]  * mat.data[10] * mat.data[15] + 
                mat.data[4]  * mat.data[11] * mat.data[14] + 
                mat.data[8]  * mat.data[6]  * mat.data[15] - 
                mat.data[8]  * mat.data[7]  * mat.data[14] - 
                mat.data[12] * mat.data[6]  * mat.data[11] + 
                mat.data[12] * mat.data[7]  * mat.data[10];

        inv[8] = mat.data[4]  * mat.data[9] * mat.data[15] - 
                mat.data[4]  * mat.data[11] * mat.data[13] - 
                mat.data[8]  * mat.data[5] * mat.data[15] + 
                mat.data[8]  * mat.data[7] * mat.data[13] + 
                mat.data[12] * mat.data[5] * mat.data[11] - 
                mat.data[12] * mat.data[7] * mat.data[9];

        inv[12] = -mat.data[4]  * mat.data[9] * mat.data[14] + 
                mat.data[4]  * mat.data[10] * mat.data[13] +
                mat.data[8]  * mat.data[5] * mat.data[14] - 
                mat.data[8]  * mat.data[6] * mat.data[13] - 
                mat.data[12] * mat.data[5] * mat.data[10] + 
                mat.data[12] * mat.data[6] * mat.data[9];

        inv[1] = -mat.data[1]  * mat.data[10] * mat.data[15] + 
                mat.data[1]  * mat.data[11] * mat.data[14] + 
                mat.data[9]  * mat.data[2] * mat.data[15] - 
                mat.data[9]  * mat.data[3] * mat.data[14] - 
                mat.data[13] * mat.data[2] * mat.data[11] + 
                mat.data[13] * mat.data[3] * mat.data[10];

        inv[5] = mat.data[0]  * mat.data[10] * mat.data[15] - 
                mat.data[0]  * mat.data[11] * mat.data[14] - 
                mat.data[8]  * mat.data[2] * mat.data[15] + 
                mat.data[8]  * mat.data[3] * mat.data[14] + 
                mat.data[12] * mat.data[2] * mat.data[11] - 
                mat.data[12] * mat.data[3] * mat.data[10];

        inv[9] = -mat.data[0]  * mat.data[9] * mat.data[15] + 
                mat.data[0]  * mat.data[11] * mat.data[13] + 
                mat.data[8]  * mat.data[1] * mat.data[15] - 
                mat.data[8]  * mat.data[3] * mat.data[13] - 
                mat.data[12] * mat.data[1] * mat.data[11] + 
                mat.data[12] * mat.data[3] * mat.data[9];

        inv[13] = mat.data[0]  * mat.data[9] * mat.data[14] - 
                mat.data[0]  * mat.data[10] * mat.data[13] - 
                mat.data[8]  * mat.data[1] * mat.data[14] + 
                mat.data[8]  * mat.data[2] * mat.data[13] + 
                mat.data[12] * mat.data[1] * mat.data[10] - 
                mat.data[12] * mat.data[2] * mat.data[9];

        inv[2] = mat.data[1]  * mat.data[6] * mat.data[15] - 
                mat.data[1]  * mat.data[7] * mat.data[14] - 
                mat.data[5]  * mat.data[2] * mat.data[15] + 
                mat.data[5]  * mat.data[3] * mat.data[14] + 
                mat.data[13] * mat.data[2] * mat.data[7] - 
                mat.data[13] * mat.data[3] * mat.data[6];

        inv[6] = -mat.data[0]  * mat.data[6] * mat.data[15] + 
                mat.data[0]  * mat.data[7] * mat.data[14] + 
                mat.data[4]  * mat.data[2] * mat.data[15] - 
                mat.data[4]  * mat.data[3] * mat.data[14] - 
                mat.data[12] * mat.data[2] * mat.data[7] + 
                mat.data[12] * mat.data[3] * mat.data[6];

        inv[10] = mat.data[0]  * mat.data[5] * mat.data[15] - 
                mat.data[0]  * mat.data[7] * mat.data[13] - 
                mat.data[4]  * mat.data[1] * mat.data[15] + 
                mat.data[4]  * mat.data[3] * mat.data[13] + 
                mat.data[12] * mat.data[1] * mat.data[7] - 
                mat.data[12] * mat.data[3] * mat.data[5];

        inv[14] = -mat.data[0]  * mat.data[5] * mat.data[14] + 
                mat.data[0]  * mat.data[6] * mat.data[13] + 
                mat.data[4]  * mat.data[1] * mat.data[14] - 
                mat.data[4]  * mat.data[2] * mat.data[13] - 
                mat.data[12] * mat.data[1] * mat.data[6] + 
                mat.data[12] * mat.data[2] * mat.data[5];

        inv[3] = -mat.data[1] * mat.data[6] * mat.data[11] + 
                mat.data[1] * mat.data[7] * mat.data[10] + 
                mat.data[5] * mat.data[2] * mat.data[11] - 
                mat.data[5] * mat.data[3] * mat.data[10] - 
                mat.data[9] * mat.data[2] * mat.data[7] + 
                mat.data[9] * mat.data[3] * mat.data[6];

        inv[7] = mat.data[0] * mat.data[6] * mat.data[11] - 
                mat.data[0] * mat.data[7] * mat.data[10] - 
                mat.data[4] * mat.data[2] * mat.data[11] + 
                mat.data[4] * mat.data[3] * mat.data[10] + 
                mat.data[8] * mat.data[2] * mat.data[7] - 
                mat.data[8] * mat.data[3] * mat.data[6];

        inv[11] = -mat.data[0] * mat.data[5] * mat.data[11] + 
                mat.data[0] * mat.data[7] * mat.data[9] + 
                mat.data[4] * mat.data[1] * mat.data[11] - 
                mat.data[4] * mat.data[3] * mat.data[9] - 
                mat.data[8] * mat.data[1] * mat.data[7] + 
                mat.data[8] * mat.data[3] * mat.data[5];

        inv[15] = mat.data[0] * mat.data[5] * mat.data[10] - 
                mat.data[0] * mat.data[6] * mat.data[9] - 
                mat.data[4] * mat.data[1] * mat.data[10] + 
                mat.data[4] * mat.data[2] * mat.data[9] + 
                mat.data[8] * mat.data[1] * mat.data[6] - 
                mat.data[8] * mat.data[2] * mat.data[5];

        det = mat.data[0] * inv[0] + mat.data[1] * inv[4] + mat.data[2] * inv[8] + mat.data[3] * inv[12];

        if (det == 0)
            throw std::invalid_argument("Determinant is zero, therefore inverse matrix doesn't exist");

        det = 1.0 / det;

        for (i = 0; i < 16; i++)
            data[i] = inv[i] * det;
    }

    Matrix4 Matrix4::inverse() const
    {
        Matrix4 result;
        result.setInverse(*this);
        return result;
    }

    Matrix4 Matrix4::inverse(const Matrix4& mat)
    {
        Matrix4 result;
        result.setInverse(mat);
        return result;
    }
    
    Matrix4 Matrix4::scale(const Matrix4& mat, const Vector3& scale)
    {
        Matrix4 result(mat);
        result.data[0] *= scale.coordinates.x;
        result.data[5] *= scale.coordinates.y;
        result.data[10] *= scale.coordinates.z;
        return result;
    }

    Matrix4 Matrix4::translate(const Matrix4& mat, const Vector3& translate)
    {
        Matrix4 result(mat);
        result.data[12] += translate.coordinates.x;
        result.data[13] += translate.coordinates.y;
        result.data[14] += translate.coordinates.z;
        return result;
    }

    Matrix4 Matrix4::perspective(const float& fieldOfView, const float& aspectRatio, const float& nearPlane, const float& farPlane)
    {
        Matrix4 result;
        Scalar yScale = 1.0 / std::tan((M_PI/ 180.0f) * fieldOfView / 2);
        Scalar xScale = yScale / aspectRatio;
        result.data[0] = xScale;
        result.data[5] = yScale;
        result.data[10] = (farPlane + nearPlane) / (nearPlane - farPlane);
        result.data[11] = -1;
        result.data[14] = 2 * farPlane * nearPlane / (nearPlane - farPlane);
        result.data[15] = 0;
        return result;
    }

    //right handed system
    Matrix4 Matrix4::lookAt(const Vector3& position, const Vector3& forward, const Vector3& up)
    {
        Vector3 zaxis = Vector3::normalize(forward - position);
        Vector3 xaxis = Vector3::normalize(Vector3::cross(zaxis, up));
        Vector3 yaxis = Vector3::cross(xaxis, zaxis);

        Matrix4 result;
        result.data[0] = xaxis.coordinates.x;
        result.data[4] = xaxis.coordinates.y;
        result.data[8] = xaxis.coordinates.z;

        result.data[1] = yaxis.coordinates.x;
        result.data[5] = yaxis.coordinates.y;
        result.data[9] = yaxis.coordinates.z;

        result.data[2] = -zaxis.coordinates.x;
        result.data[6] = -zaxis.coordinates.y;
        result.data[10] = -zaxis.coordinates.z;

        result.data[12] = -(Vector3::dot(xaxis, position));
        result.data[13] = -(Vector3::dot(yaxis, position));
        result.data[14] = (Vector3::dot(zaxis, position));

        return result;
    }

    Vector4 Matrix4::operator*(const Vector4& vec) const
    {
        return Vector4(data[0] * vec[0] + data[1] * vec[1] + data[2] * vec[2] + data[3] * vec[3],
                       data[4] * vec[0] + data[5] * vec[1] + data[6] * vec[2] + data[7] * vec[3],
                       data[8] * vec[0] + data[9] * vec[1] + data[10] * vec[2] + data[11] * vec[3],
                       data[12] * vec[0] + data[13] * vec[1] + data[14] * vec[2] + data[15] * vec[3]);
    }

    void Matrix4::print() const
    {
        for(unsigned int i = 0; i < 16; ++i)
        {
            std::cout << data[i] << " ";
            if(i == 3 || i == 7 || i == 11)
                std::cout << std::endl;
        }
    }
}