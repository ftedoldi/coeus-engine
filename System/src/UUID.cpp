#include "../UUID.hpp"

#include <random>

namespace System
{
    
    static std::random_device _randomDevice;
    static std::mt19937_64 _engine(_randomDevice());
    static std::uniform_int_distribution<std::uint64_t> _uniformDistribution;

    UUID::UUID() : _UUID(_uniformDistribution(_engine))
    {

    }

    UUID::UUID(std::uint64_t uuid) : _UUID(uuid)
    {

    }

} // namespace System
