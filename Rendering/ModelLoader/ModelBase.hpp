#ifndef __MODELBASE_H__
#define __MODELBASE_H__

#include <Component.hpp>

#include <SerializableClass.hpp>

#include <string>

namespace Odysseus
{   
    // TODO: Implement me
    class ModelBase : public System::Component
    {
        public:
            bool isPBR;

            std::string modelPath;

            std::string vertexShaderPath;
            std::string fragmentShaderPath;

            ModelBase();

            virtual void start();
            virtual void update();

            virtual void setOrderOfExecution(const short& newOrderOfExecution);

            virtual int getUniqueID();

            virtual std::string toString();

            virtual ~ModelBase() {}

            // SERIALIZABLE_CLASS(System::Component);
    };
} // namespace Odysseus


#endif // __MODELBASE_H__