#include "../../dep/include/cpptui.hpp"
#include <future>
#include <memory>

namespace MoreTUI {

template<typename Widget_tmp, typename Parent_tmp>
class Widget_c;

template<typename Widget_tmp>
class Widget_c<Widget_tmp, cpptui::ScrollableVertical> : public Widget_tmp {
    public:
        std::shared_ptr<cpptui::ScrollableVertical> parent;
        int preferred_spacing = 1;
        

        template<typename... Args_tmp>
        Widget_c(std::shared_ptr<cpptui::ScrollableVertical> parent, Args_tmp... args) 
            : parent(parent)
            , Widget_tmp(args...)
        {};


        void on_focus() override {
            Widget_tmp::on_focus();

            if (parent->height < 1) return;

            int ry = Widget_tmp::y - parent->y;
            int spacing = std::min(
                preferred_spacing, 
                (parent->height - 1) / 2
            );
            
            if (ry < spacing) {
                parent->scroll_offset += ry - spacing;
                parent->scroll_offset = std::max(parent->scroll_offset, 0);
            }
            else if (ry + Widget_tmp::height + spacing > parent->height) {
                parent->scroll_offset += spacing + ry + Widget_tmp::height - parent->height;
                parent->scroll_offset = std::min(
                    parent->scroll_offset, 
                    std::max(0, parent->content_height - parent->height)
                );
            }
        }
};

}