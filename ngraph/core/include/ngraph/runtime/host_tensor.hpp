// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "ngraph/node.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/type/element_type_traits.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            class Constant;
        }
    } // namespace op
    namespace runtime
    {
        class NGRAPH_API HostTensor : public ngraph::runtime::Tensor
        {
        public:
            HostTensor(const element::Type& element_type,
                       const Shape& shape,
                       void* memory_pointer,
                       const Strides & strides = {});
            HostTensor(const element::Type& element_type,
                       const Shape& shape)
                : HostTensor(element_type, PartialShape {shape}) {
                create(element_type, shape);
            }
            HostTensor(const element::Type& element_type,
                       const PartialShape& partial_shape); // scheme
            HostTensor();
            // explicit HostTensor(const Output<Node>&);
            // explicit HostTensor(const std::shared_ptr<op::v0::Constant>& constant);
            virtual ~HostTensor() override;

            // void initialize(const std::shared_ptr<op::v0::Constant>& constant);

            void* get_data_ptr();
            const void* get_data_ptr() const;

            void create(const element::Type& element_type, const Shape& shape) {
                // check scheme
                // TODO
            }

            operator void * () {
                return nullptr;
            }

            // /// \brief Set the shape of a node from an input
            // /// \param arg The input argument
            // void set_unary(const HostTensorPtr& arg);
            // /// \brief Set the shape of the tensor using broadcast rules
            // /// \param autob The broadcast mode
            // /// \param arg0 The first argument
            // /// \param arg1 The second argument
            // void set_broadcast(const op::AutoBroadcastSpec& autob,
            //                    const HostTensorPtr& arg0,
            //                    const HostTensorPtr& arg1);
            // /// \brief Set the shape of the tensor using broadcast rules
            // /// \param autob The broadcast mode
            // /// \param arg0 The first argument
            // /// \param arg1 The second argument
            // /// \param element_type The output element type
            // void set_broadcast(const op::AutoBroadcastSpec& autob,
            //                    const HostTensorPtr& arg0,
            //                    const HostTensorPtr& arg1,
            //                    const element::Type& element_type);

        private:
            void allocate_buffer();
            HostTensor(const HostTensor&) = delete;
            HostTensor(HostTensor&&) = delete;
            HostTensor& operator=(const HostTensor&) = delete;

            void* m_memory_pointer{nullptr};
            void* m_allocated_buffer_pool{nullptr};
            void* m_aligned_buffer_pool{nullptr};
            size_t m_buffer_size;
        };
    } // namespace runtime
} // namespace ngraph
