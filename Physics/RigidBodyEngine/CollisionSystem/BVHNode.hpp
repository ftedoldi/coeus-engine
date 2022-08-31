#ifndef BVHNODE_HPP
#define BVHNODE_HPP
#include <RigidBody.hpp>
namespace Khronos
{
    //structure that stores a potential contact that can happen
    struct PotentialContact
    {
        //array of two bodies that could be in contact and need to be checked
        RigidBody* body[2];
    };

    //class for a node in the bounding volume hierarchy (BVH)
    //since we can have more than one type of bounding volume (sphere, cube, ecc.)
    //we create a template class, specifing the bounding volume type we'll use.
    template<class BoundingVolumeClass>
    class BVHNode
    {
    public:
        /**
         * We want to have a binary tree as hierarchy because its more convinient.
         * It can be better in term of speed of execution, its easier to implement
         * and simplify several algorithms we'll see later
        */
        
        // Holds the children of a node
        BVHNode* children[2];

        // Holds a single bounding volume encompassing all the descendents of this node
        BoundingVolumeClass volume;

        // Holds the rigid body at this node of hierarchy.
        // Only leaf nodes can have a rigid body defined.
        RigidBody* body;

        // Holds the parent node of this node
        BVHNode* parent;

        // Constructor that creates a new node in the hierarchy from the given parameters
        BVHNode(BVHNode* parent, const BoundingVolumeClass& volume, RigidBody* body = nullptr);

        // Check if this node is at the bottom of the hierarchy
        bool isLeaf() const;

        /**
         * Check the potential contacts from this node, downward in the hierarchy
         * and writes them to the given array (up to the given limit).
         * Returns the number of potential contacts it found.
         * We can also encounter some false positives.
        */
        unsigned int getPotentialContacts(PotentialContact* contacts, unsigned int limit) const;

        // Insert the given rigid body with the given bounding volume into the hierarchy.
        void insert(RigidBody* body, const BoundingVolumeClass& volume);

        /**
         * Deletes this node, removing it from the hierarcy, along with its rigid body and child nodes.
         * This method also delete all the siblings of this node, and change the parent node so that it
         * contains the data of that sibling.
         * It also forces the hierarchy above the current node to reconsider its bounding volume.
        */
        ~BVHNode();
    
    protected:
        
        // Check if this volume overlaps with other volume
        // Note that each BoundingVolumeClass must implement the overlaps method
        bool overlaps(const BVHNode<BoundingVolumeClass>* other) const;

        // Check the potential contacts between this node and the given other node
        // writing them to the given array.
        // Returns the number of potential contacts found.
        unsigned int getPotentialContactsWith(const BVHNode<BoundingVolumeClass>* other, PotentialContact* contacts, unsigned int limit) const;

        // For non leaf nodes, this method recalculates the bounding volume of the corresponding node,
        // based on the bounting volumes of it's children
        void recalculateBoundingVolume(bool recurse = true);
    };
}

#endif