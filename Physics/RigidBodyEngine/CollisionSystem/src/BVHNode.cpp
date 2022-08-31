#include "../BVHNode.hpp"

namespace Khronos
{
    template<class BoundingVolumeClass>
    BVHNode<BoundingVolumeClass>::BVHNode(BVHNode* parent, const BoundingVolumeClass& volume, RigidBody* body)
    : parent(parent), volume(volume), body(body)
    {
        children[0] = children[1] = nullptr;
    }


    template<class BoundingVolumeClass>
    bool BVHNode<BoundingVolumeClass>::isLeaf() const
    {
        return (body != nullptr);
    }

    template<class BoundingVolumeClass>
    bool BVHNode<BoundingVolumeClass>::overlaps(const BVHNode<BoundingVolumeClass>* other) const
    {
        return this->volume->overlaps(other->volume);
    }

    template<class BoundingVolumeClass>
    unsigned int BVHNode<BoundingVolumeClass>::getPotentialContacts(PotentialContact* contacts, unsigned int limit) const
    {
        // If we are at leaf node (so we dont have children) and we don't have room for contacts (limit == 0) we leave
        if(this->isLeaf() || limit == 0)
            return 0;
        
        // Get the potential contacts of one of our children with the other one
        return this->children[0]->getPotentialContactsWith(this->children[1], contacts, limit);
    }

    template<class BoundingVolumeClass>
    unsigned int BVHNode<BoundingVolumeClass>::getPotentialContactsWith(const BVHNode<BoundingVolumeClass>* other, 
                                                                        PotentialContact* contacts, 
                                                                        unsigned int limit) const
    {
        // If the two volumes don't overlaps or we don't have room for contacts we leave
        if(!this->overlaps(other) || limit == 0)
            return 0;

        // If the two nodes are both leaf nodes, we have a potential contact
        if(this->isLeaf() && other->isLeaf())
        {
            contacts->body[0] = this->body;
            contacts->body[1] = other->body;
            return 1;
        }

        /**
         * Now determine which node to descend into.
         * If one of the two nodes is a leaf, we descend the other.
         * If both are branches, we descend the one with larger size.
        */
       // In this case we want to descend this node
       if(other->isLeaf() || (!this->isLeaf() && this->volume->getSize() >= other->volume->getSize()))
       {
            // Recurse into this
            unsigned int count = this->children[0]->getPotentialContactsWith(other, contacts, limit);

            // Check if we can recurse the other side too
            if(limit > count)
            {
                return count + this->children[1]->getPotentialContactsWith(other, contacts + count, limit - count);
            }else
            {
                return count;
            }
       }
       else
       {
            // Recurse into other node
            unsigned int count = getPotentialContactsWith(other->children[0], contacts, limit);

            // Check if we can recurse the other side too
            if(limit > count)
            {
                return count + getPotentialContactsWith(other->children[1], contacts + count, limit - count);
            }
            else
            {
                return count;
            }
       }
    }

    template<class BoundingVolumeClass>
    void BVHNode<BoundingVolumeClass>::recalculateBoundingVolume(bool recurse)
    {
        if(this->isLeaf())
            return;

        // We call the constructor that allow us to create a bounding volume that encapsulate the 
        // two children bounding volume, based on their volume
        this->volume = BoundingVolumeClass(this->children[0]->volume, this->children[1]->volume);

        // Recurse up to the tree to update parent's bounding volume
        if(parent)
            parent->recalculateBoundingVolume(true);
    }

    template<class BoundingVolumeClass>
    void BVHNode<BoundingVolumeClass>::insert(RigidBody* newBody, const BoundingVolumeClass& newVolume)
    {
        // If this node is a leaf, the only option is to create two new children and place the
        // rigid body inside one.
        if(this->isLeaf())
        {
            // The first children is a copy of this since this node will now encapsulate the two children
            this->children[0] = new BVHNode<class BoundingVolumeClass>(this, this->volume, this->body);
            // The second children is the new rigid body
            this->children[1] = new BVHNode<class BoundingVolumeClass>(this, newVolume, newBody);

            // The body of this is now null since it is no more a leaf
            this->body = nullptr;

            // We then need to recalculate the bounding volume
            this->recalculateBoundingVolume();
        }
        // Otherwise we need to work out which child gets to keep the new inserted body.
        // We give it to whoever would grow the least to incorporate it.
        else
        {
            if(children[0]->volume.getGrowth(newVolume) < children[1]->volume.getGrowth(newVolume))
            {
                children[0]->insert(newBody, newVolume);
            }
            else
            {
                children[1]->insert(newBody, newVolume);
            }
        }
    }

    template<class BoundingVolumeClass>
    BVHNode<BoundingVolumeClass>::~BVHNode()
    {
        if(parent)
        {
            BVHNode<BoundingVolumeClass>* sibling;
            if(parent->children[0] == this)
                sibling = parent->children[1];
            else
                sibling = parent->children[0];

            // Write sibling data to the parent
            parent->volume = sibling->volume;
            parent->body = sibling->body;
            parent->children[0] = sibling->children[0];
            parent->children[1] = sibling->children[1];

            // Delete the sibling
            sibling->parent = nullptr;
            sibling->body = nullptr;
            sibling->children[0] = nullptr;
            sibling->children[1] = nullptr;
            delete sibling;

            // Recalculate the parent's bounding volume
            parent->recalculateBoundingVolume();
        }

        // Delete our children since the need to be deleted and are not 
        // useful anymore
        if(children[0])
        {
            children[0]->parent = nullptr;
            delete children[0];
        }

        if(children[1])
        {
            children[1]->parent = nullptr;
            delete children[1];
        }
    }
}