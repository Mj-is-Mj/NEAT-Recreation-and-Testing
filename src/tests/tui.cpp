#include "../inc/moretui.hpp"
#include <cstdlib>
#include <memory>

using namespace MoreTUI;


class MyRoot : public cpptui::Vertical {
    public: 
        bool on_event(const cpptui::Event& event) override {
            if (cpptui::Vertical::on_event(event)) return true;
            if (event.type == cpptui::EventType::Key && event.key == 'q') {
                cpptui::App::quit();
                return true;
            }
            return false;
        }
};

int main() {
    cpptui::App app;
    auto root = std::make_shared<MyRoot>();


    auto uhhh_pane = std::make_shared<cpptui::ScrollableVertical>();
    for (size_t i = 0; i < 50; ++i) {
        using UhhhButton = Widget_c<cpptui::Button, cpptui::ScrollableVertical>;
        // Make title
        char title_buff[24];
        sprintf(title_buff, "Uhhh... %ld", i);

        // Make button
        auto uhhh_button = std::make_shared<UhhhButton>(uhhh_pane, title_buff);
        uhhh_button->fixed_width = 12;

        uhhh_pane->add(uhhh_button);
    }

    root->add(uhhh_pane);
    app.run(root);

    return EXIT_SUCCESS;
}