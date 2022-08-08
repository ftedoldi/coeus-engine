#ifndef __PIXELDATA_H__
#define __PIXELDATA_H__

namespace System
{
    struct PixelData {
        float objectID;
        float drawID;
        float primitiveID;

        PixelData() {
            objectID = 0.0f;
            drawID = 0.0f;
            primitiveID = 0.0f;
        }
    };  
} // namespace System


#endif // __PIXELDATA_H__