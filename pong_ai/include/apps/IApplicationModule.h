#pragma once
namespace pong_ai {
    namespace apps {

        class IApplicationModule {
        public:
            virtual void train() = 0;
            virtual void test() = 0;
            virtual ~IApplicationModule() = default;
        };

    } }
