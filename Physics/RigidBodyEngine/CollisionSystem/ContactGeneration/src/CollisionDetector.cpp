#include "../CollisionDetector.hpp"

namespace Khronos
{
    void CollisionDetector::sphereAndSphere(const CollisionSphere* firstSphere, const CollisionSphere* secondSphere, CollisionData* data)
    {
        // Check if we have contacts
        if(data->contactsLeft <= 0)
            return;

        if(!IntersectionTests::sphereAndSphere(firstSphere, secondSphere))
            return;

        // Retrieve the sphere positions based on their transform matrix (getting last column)
        Athena::Vector3 firstSpherePosition = firstSphere->getAxis(3);
        Athena::Vector3 secondSpherePosition = secondSphere->getAxis(3);

        // Find the vector between the spheres and get its magnitude
        Athena::Vector3 midline = firstSpherePosition - secondSpherePosition;
        Athena::Scalar size = midline.magnitude();
        
        if(size <= 0.0f || size >= firstSphere->radius + secondSphere->radius)
            return;

        // The contact normal is given by the normal of the first face
        Athena::Vector3 normal = midline * (((Athena::Scalar)1.0) / size);

        auto contact = data->getContact();
        contact->contactNormal = normal;
        // Arbitrary point
        contact->contactPoint = firstSpherePosition + midline * Athena::Scalar(0.5);
        // Size of the maximum penetration
        contact->penetration = firstSphere->radius + secondSphere->radius - size;

        contact->body[0] = firstSphere->body;
        contact->body[1] = secondSphere->body;
        contact->friction = data->friction;
        contact->restitution = data->restitution;
    }

    void CollisionDetector::sphereAndHalfSpace(const CollisionSphere* sphere, const CollisionPlane* plane, CollisionData* data)
    {
        if(data->contactsLeft <= 0)
            return;

        if(!IntersectionTests::sphereAndHalfSpace(sphere, plane))
            return;

        Athena::Vector3 spherePosition = sphere->getAxis(3);

        // Find the distance by the sphere from the plane
        Athena::Scalar sphereDistance = plane->direction.dot(spherePosition) - sphere->radius - plane->offset;

        // The sphere and the plane didn't collide
        if(sphereDistance >= 0)
            return;

        auto contact = data->getContact();
        contact->contactNormal = plane->direction;
        contact->penetration = -sphereDistance;
        contact->contactPoint = spherePosition - plane->direction * (sphereDistance + sphere->radius);


        contact->body[0] = sphere->body;
        contact->body[1] = nullptr;
        contact->friction = data->friction;
        contact->restitution = data->restitution;
    }

    void CollisionDetector::boxAndHalfSpace(const CollisionBox* box, const CollisionPlane* plane, CollisionData* data)
    {
        if(data->contactsLeft <= 0)
            return;
        
        if(!IntersectionTests::boxAndHalfSpace(box, plane))
            return;
        
        // Since the intersection test passed, we have an intersection
        // so we need to find the intersection points
        static Athena::Scalar mults[8][3] = {{1, 1, 1},{-1, 1, 1},{1, -1, 1},{-1, -1, 1},
                               {1, 1, -1},{-1, 1, -1},{1, -1, -1},{-1, -1, -1}};
        
        for(unsigned int i = 0; i < 8; ++i)
        {
            // Calculate the position of each vertex
            Athena::Vector3 vertexPos(mults[i][0], mults[i][1], mults[i][2]);
            vertexPos.coordinates.x *= box->halfSize.coordinates.x;
            vertexPos.coordinates.y *= box->halfSize.coordinates.y;
            vertexPos.coordinates.z *= box->halfSize.coordinates.z;
            Athena::Vector4 vertexPos4(vertexPos.coordinates.x, vertexPos.coordinates.y, vertexPos.coordinates.z, 1);
            vertexPos = (box->transform * vertexPos4).xyz();

            // Calculate the distance of the vertex from the plane
            Athena::Scalar vertexDistance = vertexPos * plane->direction;

            if(vertexDistance <= plane->offset)
            {
                auto contact = data->getContact();
                contact->contactPoint = plane->direction;
                contact->contactPoint *= (vertexDistance - plane->offset);
                contact->contactPoint += vertexPos;
                contact->contactNormal = plane->direction;
                contact->penetration = plane->offset - vertexDistance;
                std::cout << contact->penetration << std::endl;

                contact->body[0] = box->body;
                // Since the plane has no rigid body
                contact->body[1] = nullptr;
                contact->friction = data->friction;
                contact->restitution = data->restitution;

                if(!data->hasContactsLeft())
                    return;
            }
        }
    }

    void CollisionDetector::boxAndSphere(const CollisionBox* box, const CollisionSphere* sphere, CollisionData* data)
    {
        /**
         * Firstly we need to transform the center of the sphere into the box coordinates
         * to be able to do next calculus
        */
       Athena::Vector3 center = sphere->getAxis(3);
       Athena::Vector3 relativeSphereCenter = box->transform.transformInverse(center);

       /**
        * Check if we can exclude contacts by the idea of separating axes.
        * If we can find any direction in space in which two objects are
        * not colliding, then the two objects are not colliding at all.
       */
      if(Athena::Math::scalarAbs(relativeSphereCenter.coordinates.x) - sphere->radius > box->halfSize.coordinates.x ||
         Athena::Math::scalarAbs(relativeSphereCenter.coordinates.y) - sphere->radius > box->halfSize.coordinates.y ||
         Athena::Math::scalarAbs(relativeSphereCenter.coordinates.z) - sphere->radius > box->halfSize.coordinates.z)
        {
            return;
        }

        /** 
         * Now we need to find the closest point in the box to the center of the sphere
         * so we can test if the distance between the closest point of the box and the
         * center of the sphere is less than the radius of the sphere.
         * If so the two objects are touching
        */
        Athena::Vector3 closestPoint(0.0, 0.0, 0.0);
        Athena::Scalar distance;

        // Now we check for the closest point for each axis
        distance = relativeSphereCenter.coordinates.x;
        if(distance > box->halfSize.coordinates.x) distance = box->halfSize.coordinates.x;
        if(distance < -box->halfSize.coordinates.x) distance = -box->halfSize.coordinates.x;
        closestPoint.coordinates.x = distance;

        distance = relativeSphereCenter.coordinates.y;
        if(distance > box->halfSize.coordinates.y) distance = box->halfSize.coordinates.y;
        if(distance < -box->halfSize.coordinates.y) distance = -box->halfSize.coordinates.y;
        closestPoint.coordinates.y = distance;

        distance = relativeSphereCenter.coordinates.z;
        if(distance > box->halfSize.coordinates.z) distance = box->halfSize.coordinates.z;
        if(distance < -box->halfSize.coordinates.z) distance = -box->halfSize.coordinates.z;
        closestPoint.coordinates.z = distance;

        // Now perform the check if we are in contact
        distance = (closestPoint - relativeSphereCenter).squareMagnitude();
        if(distance > sphere->radius * sphere->radius)
            return;
        
        /**
         * Since the contact properties need to be given in world coordinates
         * we transform the closest point in world coordinates by transforming
         * the point generated earlier
        */
       Athena::Vector3 closestPointWS = box->transform.transform(closestPoint);

       auto contact = data->getContact();
       contact->contactNormal = closestPoint - center;
       contact->contactNormal.normalize();
       contact->contactPoint = closestPointWS;
       contact->penetration = sphere->radius - Athena::Math::scalarSqrt(distance);

       contact->body[0] = box->body;
       contact->body[1] = sphere->body;
       contact->friction = data->friction;
       contact->restitution = data->restitution;
    }

    void CollisionDetector::boxAndPoint(const CollisionBox& box, const Athena::Vector3& point, CollisionData* data)
    {
        // Transform the point into box coordinates
        Athena::Vector3 relativePointPosition = box.transform.transformInverse(point);

        Athena::Vector3 normal;

        // We perform a check for each axis, looking for the axis
        // on which the penetration is least deep

        Athena::Scalar min_depth = box.halfSize.coordinates.x - Athena::Math::scalarAbs(relativePointPosition.coordinates.x);
        // The point is not interpenetrated, we don't have a contact
        if(min_depth < 0)
            return;
        // The normal is in the direction of where the point is situated
        normal = box.getAxis(0) * ((relativePointPosition.coordinates.x < 0) ? -1 : 1);

        Athena::Scalar depth = box.halfSize.coordinates.y - Athena::Math::scalarAbs(relativePointPosition.coordinates.y);
        if(depth < 0)
        {
            return;
        }else if(depth < min_depth)
        {
            min_depth = depth;
            normal = box.getAxis(1) * ((relativePointPosition.coordinates.y < 0) ? -1 : 1);
        }

        depth = box.halfSize.coordinates.z - Athena::Math::scalarAbs(relativePointPosition.coordinates.z);
        if(depth < 0)
        {
            return;
        }else if(depth < min_depth)
        {
            min_depth = depth;
            normal = box.getAxis(2) * ((relativePointPosition.coordinates.z < 0) ? -1 : 1);
        }

        auto contact = data->getContact();
        contact->contactNormal = normal;
        contact->contactPoint = point;
        contact->penetration = min_depth;

        contact->body[0] = box.body;
        // The rigid body associated to the point is null since we don't know
        // at which body the point is associated
        contact->body[1] = nullptr;
        contact->friction = data->friction;
        contact->restitution = data->restitution;
    }
}