#include "../DummyComponent.hpp"

    DummyComponent::DummyComponent()
    {
        
    }

    void DummyComponent::start()
    {
        std::cout << "Dummy component Created!";
    }

    void DummyComponent::update()
    {
        std:: cout << "Dummy component is running!";
    }

    std::string DummyComponent::getUniqueID()
    {
        return "DummyComponent";
    }

    std::string DummyComponent::toString()
    {
        return "DummyComponent";
    }

    DummyComponent::~DummyComponent()
    {

    }
