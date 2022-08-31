#ifndef COLLISIONUTILITY_HPP
#define COLLISIONUTILITY_HPP
//#include <CollisionBox.hpp>

namespace Khronos
{
    class CollisionBox;
    class CollisionUtility
    {
        public:

            /**
             * Two objects cannot be in contact as long as there is some axis on which the objects
             * can be projected where they are not in contact.
             * With this method we project the object onto the axis in order to check if the projections
             * are overlapping.
             * If they don't overlap than we have no collision.
             * This acts as an early out.
            */
            static inline Athena::Scalar transformToAxis(const CollisionBox& box, const Athena::Vector3 &axis)
            {
                return 
                    box.halfSize.coordinates.x * Athena::Math::scalarAbs(axis * box.getAxis(0)) +
                    box.halfSize.coordinates.y * Athena::Math::scalarAbs(axis * box.getAxis(1)) +
                    box.halfSize.coordinates.z * Athena::Math::scalarAbs(axis * box.getAxis(2));
            }

            /**
             * This function checks if the two boxes are overlapping along
             * the given axis.
             * The parameter toCentre is used to pass in the vector between the 
             * boxes center points.
            */
            static inline bool overlapOnAxis(const CollisionBox& firstBox,
                                             const CollisionBox& secondBox,
                                             const Athena::Vector3& axis,
                                             const Athena::Vector3& toCentre)
            {
                Athena::Scalar firstBoxProjection = transformToAxis(firstBox, axis);
                Athena::Scalar secondBoxProjection = transformToAxis(secondBox, axis);

                Athena::Scalar distance = Athena::Math::scalarAbs(toCentre * axis);

                // Check for overlap
                // Pag 313 image shows why this works
                return (distance < firstBoxProjection + secondBoxProjection);
            }
    };
}

#endif