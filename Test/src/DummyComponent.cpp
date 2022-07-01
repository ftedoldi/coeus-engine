#include "../DummyComponent.hpp"

    DummyComponent::DummyComponent()
    {
        var = 10;
    }

    void DummyComponent::start()
    {
        std::cout << "Dummy component Created!";
    }

    void DummyComponent::update()
    {
        std::cout << "Dummy component is running!";
    }

    void DummyComponent::setOrderOfExecution(const short& newOrderOfExecution)
    {
        _orderOfExecution = newOrderOfExecution;
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
