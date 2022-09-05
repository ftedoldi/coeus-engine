#include "../ParticleForceRegistry.hpp"

namespace Khronos
{
    void ParticleForceRegistry::add(Particle* particle, ParticleForceGenerator* pfg)
    {
        ParticleForceRegistry::ParticleForceRegistration reg;
        reg.particle = particle;
        reg.pfg = pfg;
        this->registrations.push_back(reg);
    }

    void ParticleForceRegistry::remove(Particle* particle, ParticleForceGenerator* pfg)
    {
        ParticleForceRegistry::ParticleForceRegistration reg;
        reg.particle = particle;
        reg.pfg = pfg;
        
        Registry::iterator i = this->registrations.begin();
        for(; i != this->registrations.end(); ++i)
        {
            if(reg.particle == particle && reg.pfg == pfg)
            {
                this->registrations.erase(i);
                return;
            }
        }
        std::cout << "No registration found to delete" << std::endl;
        return;
    }

    void ParticleForceRegistry::clear()
    {
        this->registrations.clear();
    }

    void ParticleForceRegistry::updateForces(Athena::Scalar dt)
    {
        Registry::iterator i = this->registrations.begin();
        for(; i != this->registrations.end(); ++i)
        {
            i->pfg->updateForce(i->particle, dt);
        }
    }
}