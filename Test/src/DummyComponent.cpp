#include "../DummyComponent.hpp"

    DummyComponent::DummyComponent()
    {
        // _showComponentInEditor = true;
        var = 10;
        std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++";
    }

    void DummyComponent::start()
    {
        std::cout << "Dummy component Created!";
    }

    void DummyComponent::update()
    {
        // std::cout << "Dummy component is running!";
    }

    void DummyComponent::setOrderOfExecution(const short& newOrderOfExecution)
    {
        _orderOfExecution = newOrderOfExecution;
    }

    int DummyComponent::getUniqueID()
    {
        return _uniqueID;
    }

    std::string DummyComponent::toString()
    {
        return "DummyComponent";
    }

    DummyComponent::~DummyComponent()
    {

    }
