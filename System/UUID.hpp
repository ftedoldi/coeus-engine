#ifndef __UUID_H__
#define __UUID_H__

#include <xhash>
#include <cinttypes>

namespace System {
    class UUID {
        private:
            std::uint64_t _UUID;
        
        public:
            UUID();
            UUID(std::uint64_t uuid);

            operator std::uint64_t() const { return _UUID; }
    };
}

namespace std {
    // template specialization -> we override one of the std library class in order to be able to hast my class
    template<>
    struct hash<System::UUID> 
    {
        std::size_t operator() (const System::UUID& uuid) const {
            return hash<uint64_t>()((uint64_t)uuid);
        }
    };
}

#endif // __UUID_H__