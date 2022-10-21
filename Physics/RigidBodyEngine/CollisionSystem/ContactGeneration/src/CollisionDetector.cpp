#include "../CollisionDetector.hpp"
#include <CollisionUtility.hpp>

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
       if(!data->hasContactsLeft())
        return;
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
        Athena::Vector3 closestPoint;
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
       contact->contactNormal = closestPointWS - center;
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

    static inline bool TryAxis(const CollisionBox* box1, const CollisionBox* box2,
                               Athena::Vector3& axis, Athena::Vector3& toCenter,
                               unsigned int index, Athena::Scalar& smallestPenetration,
                               int& smallestCase)
    {
        if(axis.squareMagnitude() < 0.0001)
            return true;
        axis.normalize();

        Athena::Scalar penetration = CollisionUtility::penetrationOnAxis(box1, box2, axis, toCenter);

        if(penetration < 0)
            return false;
        
        if(penetration < smallestPenetration)
        {
            smallestPenetration = penetration;
            smallestCase = index;
        }
        return true;
    }

    void CollisionDetector::fillPointFaceBoxBox(const CollisionBox* box1, const CollisionBox* box2,
                                                const Athena::Vector3& toCenter, CollisionData* data,
                                                unsigned int best, Athena::Scalar penetration)
    {
        // This method is called when a vertex from box2 is in contact with box1
        auto contact = data->getContact();

        Athena::Vector3 normal = box1->getAxis(best);
        if(Athena::Vector3::dot(box1->getAxis(best), toCenter) > 0)
            normal = normal * -1.0f;

        // Work out which vertex of box2 we're colliding with
        Athena::Vector3 vertex = box2->halfSize;
        if(Athena::Vector3::dot(box2->getAxis(0), normal) < 0)
            vertex.coordinates.x = -vertex.coordinates.x;
        
        if(Athena::Vector3::dot(box2->getAxis(1), normal) < 0)
            vertex.coordinates.y = -vertex.coordinates.y;
        
        if(Athena::Vector3::dot(box2->getAxis(2), normal) < 0)
            vertex.coordinates.z = -vertex.coordinates.z;

        contact->contactNormal = normal;
        contact->penetration = penetration;
        contact->contactPoint = box2->transform.transform(vertex);

        //contact->contactPoint.print();

        contact->body[0] = box1->body;
        contact->body[1] = box2->body;
        contact->friction = data->friction;
        contact->restitution = data->restitution;

    }

    Athena::Vector3 CollisionDetector::contactPoint(const Athena::Vector3& pOne, const Athena::Vector3& dOne,
                                    Athena::Scalar oneSize, const Athena::Vector3& pTwo, const Athena::Vector3& dTwo,
                                    Athena::Scalar twoSize, bool useOne)
    {
        // If useOne is true, and the contact point is outside
        // the edge (in the case of an edge-face contact) then
        // we use one's midpoint, otherwise we use two's.

        Athena::Vector3 toSt, cOne, cTwo;
        Athena::Scalar dpStaOne, dpStaTwo, dpOneTwo, smOne, smTwo;
        Athena::Scalar denom, mua, mub;

        smOne = dOne.squareMagnitude();
        smTwo = dTwo.squareMagnitude();
        dpOneTwo = Athena::Vector3::dot(dTwo, dOne);

        toSt = pOne - pTwo;
        dpStaOne = Athena::Vector3::dot(dOne, toSt);
        dpStaTwo = Athena::Vector3::dot(dTwo, toSt);

        denom = smOne * smTwo - dpOneTwo * dpOneTwo;

        if(Athena::Math::scalarAbs(denom) < 0.00001)
            return useOne ? pOne : pTwo;
        
        mua = (dpOneTwo * dpStaTwo - smTwo * dpStaOne) / denom;
        mub = (smOne * dpStaTwo - dpOneTwo * dpStaOne) / denom;

        if(mua > oneSize || mua < -oneSize || mub > twoSize || mub < -twoSize)
            return useOne ? pOne : pTwo;
        else
        {
            cOne = pOne + dOne * mua;
            cTwo = pTwo + dTwo * mub;
            return cOne * 0.5 + cTwo * 0.5;
        }
    }


    void CollisionDetector::boxAndBox(const CollisionBox* box1, const CollisionBox* box2, CollisionData* data)
    {
        if(!data->hasContactsLeft())
            return;

        //if(!IntersectionTests::boxAndBox(box1, box2))
        //    return;
        
        // Vector between the two boxes centers
        Athena::Vector3 toCenter = box2->getAxis(3) - box1->getAxis(3);

        Athena::Scalar penetration = SCALAR_MAX;
        int best = 0xffffff;

        // We check each axis, if the projections of the boxes on at least one
        // axis doesn't intersect, we don't have a collision, otherwise, we can have it
        // and we keep track of the axis with the smallest penetration
        if(!TryAxis(box1, box2, box1->getAxis(0), toCenter, 0, penetration, best)) return;
        if(!TryAxis(box1, box2, box1->getAxis(1), toCenter, 1, penetration, best)) return;
        if(!TryAxis(box1, box2, box1->getAxis(2), toCenter, 2, penetration, best)) return;

        if(!TryAxis(box1, box2, box2->getAxis(0), toCenter, 3, penetration, best)) return;
        if(!TryAxis(box1, box2, box2->getAxis(1), toCenter, 4, penetration, best)) return;
        if(!TryAxis(box1, box2, box2->getAxis(2), toCenter, 5, penetration, best)) return;

        int bestSingleAxis = best;

        if(!TryAxis(box1, box2, Athena::Vector3::cross(box1->getAxis(0), box2->getAxis(0)), toCenter, 6, penetration, best)) return;
        if(!TryAxis(box1, box2, Athena::Vector3::cross(box1->getAxis(0), box2->getAxis(1)), toCenter, 7, penetration, best)) return;
        if(!TryAxis(box1, box2, Athena::Vector3::cross(box1->getAxis(0), box2->getAxis(2)), toCenter, 8, penetration, best)) return;
        if(!TryAxis(box1, box2, Athena::Vector3::cross(box1->getAxis(1), box2->getAxis(0)), toCenter, 9, penetration, best)) return;
        if(!TryAxis(box1, box2, Athena::Vector3::cross(box1->getAxis(1), box2->getAxis(1)), toCenter, 10, penetration, best)) return;
        if(!TryAxis(box1, box2, Athena::Vector3::cross(box1->getAxis(1), box2->getAxis(2)), toCenter, 11, penetration, best)) return;
        if(!TryAxis(box1, box2, Athena::Vector3::cross(box1->getAxis(2), box2->getAxis(0)), toCenter, 12, penetration, best)) return;
        if(!TryAxis(box1, box2, Athena::Vector3::cross(box1->getAxis(2), box2->getAxis(1)), toCenter, 13, penetration, best)) return;
        if(!TryAxis(box1, box2, Athena::Vector3::cross(box1->getAxis(2), box2->getAxis(2)), toCenter, 14, penetration, best)) return;

        assert(best != 0xffffff);
        
        if(best < 3)
        {
            fillPointFaceBoxBox(box1, box2, toCenter, data, best, penetration);
        }
        else if (best < 6)
        {
            fillPointFaceBoxBox(box2, box1, toCenter * -1.0f, data, best - 3, penetration);
        }
        else
        {
            best -=6;
            int box1AxisIndex = best / 3;
            int box2AxisIndex = best % 3;
            Athena::Vector3 box1Axis = box1->getAxis(box1AxisIndex);
            Athena::Vector3 box2Axis = box2->getAxis(box2AxisIndex);
            Athena::Vector3 axis = Athena::Vector3::cross(box1Axis, box2Axis);
            axis.normalize();

            if(Athena::Vector3::dot(axis, toCenter) > 0)
                axis *= -1.0f;
            
            Athena::Vector3 ptOnBox1Edge = box1->halfSize;
            Athena::Vector3 ptOnBox2Edge = box2->halfSize;

            for(unsigned int i = 0; i < 3; ++i)
            {
                if(i == box1AxisIndex) ptOnBox1Edge[i] = 0;
                else if(Athena::Vector3::dot(box1->getAxis(i), axis) > 0) ptOnBox1Edge[i] = -ptOnBox1Edge[i];

                if(i == box2AxisIndex) ptOnBox2Edge[i] = 0;
                else if(Athena::Vector3::dot(box2->getAxis(i), axis) > 0) ptOnBox2Edge[i] = -ptOnBox2Edge[i];
            }

            ptOnBox1Edge = box1->transform.transform(ptOnBox1Edge);
            ptOnBox2Edge = box2->transform.transform(ptOnBox2Edge);

            Athena::Vector3 vertex = contactPoint(
                ptOnBox1Edge, box1Axis, box1->halfSize[box1AxisIndex],
                ptOnBox2Edge, box2Axis, box2->halfSize[box2AxisIndex],
                bestSingleAxis > 2
            );

            auto contact = data->getContact();

            contact->penetration = penetration;
            contact->contactNormal = axis;
            contact->contactPoint = vertex;

            contact->body[0] = box1->body;
            contact->body[1] = box2->body;
            contact->friction = data->friction;
            contact->restitution = data->restitution;

        }

    }
}