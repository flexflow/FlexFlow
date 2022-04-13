#include "flexflow/substitution_loader.h"
#include <cassert>
#include <functional>

namespace FlexFlow::substitution_loader {

void from_json(json const &j, Parameter &p) {
    j.at("key").get_to(p.key);
    j.at("value").get_to(p.value);
}

void from_json(json const &j, Tensor &t) {
    j.at("opId").get_to(t.opId);
    j.at("tsId").get_to(t.tsId);
}

tl::optional<int> Operator::at(PMParameter key) const {
    tl::optional<int> value = tl::nullopt;
    for (Parameter const &p : this->para) {
        if (p.key == key) {
            assert (!value.has_value());
            value = p.key;
        }
    }

    return value;
}

void from_json(json const &j, Operator &o) {
    j.at("type").get_to(o.op_type);
    j.at("input").get_to(o.input);
    j.at("para").get_to(o.para);
}

void from_json(json const &j, MapOutput &m) {
    j.at("dstOpId").get_to(m.dstOpId);
    j.at("dstTsId").get_to(m.dstTsId);
    j.at("srcOpId").get_to(m.srcOpId);
    j.at("srcTsId").get_to(m.srcTsId);
}

void from_json(json const &j, Rule &r) {
    j.at("srcOp").get_to(r.srcOp);
    j.at("dstOp").get_to(r.dstOp);
    j.at("mappedOutput").get_to(r.mappedOutput);
}

void from_json(json const &j, RuleCollection &c) {
    j.at("rule").get_to(c.rules);
}

RuleCollection load_rule_collection(std::istream &s) {
    json j; 
    s >> j;
    RuleCollection rule_collection = j;
    return rule_collection;
}

RuleCollection load_rule_collection_from_path(std::string const &path) {
    std::ifstream input(path);
    return load_rule_collection(input);
}

int get_num_outputs(Operator const &op) {
    switch (op.op_type) {
        case OP_SPLIT:
            return op.at(PM_NUM_OUTPUTS).value();
        default:
            return 1;
    }
}

int get_num_inputs(Operator const &op) {
    switch (op.op_type) {
        case OP_EW_ADD: // binary ops
        case OP_EW_SUB:
        case OP_EW_MUL:
        case OP_EW_DIV:
        case OP_EW_EQUAL:
        case OP_EW_GREATER:
        case OP_EW_LESS:
        case OP_EW_MAX:
        case OP_EW_MIN:
            return 2;
        case OP_LINEAR:
            return 1;
        case OP_RELU: 
        case OP_IDENTITY:
        case OP_SIGMOID:
        case OP_TANH:
        case OP_ELU:
            return 1;
        case OP_CONCAT:
            return op.at(PM_NUM_INPUTS).value();
        default:
            json j = op.op_type;
            std::string s = j;
            throw std::runtime_error("Unknown num_inputs for operator " + s);
    }
}

OpX *create_opx(Operator const &op, TensorX const &input1, TensorX const &input2, TensorX const &input3, TensorX const &input4) {
    int num_inputs = get_num_inputs(op);
    int num_outputs = get_num_outputs(op);
    
    OpX *opx = new OpX(op.op_type, num_inputs, num_outputs, input1, input2, input3, input4);
    for (Parameter const &p : op.para) {
        opx->add_pm_constraint(COMPARE_EQ, p.key, p.value);
    }

    return opx;
}

std::vector<OpX *> create_rule_graph(std::vector<Operator> const &ops, std::function<TensorX(int, int)> const &get_input_tensor) {
    std::vector<OpX *> rule_graph;

    for (int i = 0; i < ops.size(); i++) {
        Operator const &op = ops[i];
        std::array<TensorX, 4> inputs;
        std::fill(inputs.begin(), inputs.end(), TensorX::NO_TX);

        for (int j = 0; j < op.input.size(); j++) {
            int opId = op.input[j].opId;
            int tsId = op.input[j].tsId;
            if (opId < 0) {
                inputs[j] = get_input_tensor(opId, tsId);
            } else {
                inputs[j] = rule_graph[opId]->outputs[tsId];
            }
        }

        OpX *opx = create_opx(ops[i], inputs[0], inputs[1], inputs[2], inputs[3]);
        rule_graph.push_back(opx);
    }

    return rule_graph;
}

void create_xfer(GraphXfer &xfer, Rule const &r) {
    std::unordered_map<std::pair<int, int>, TensorX> input_tensors;
    std::function<TensorX(int,int)> get_input_tensor = [&xfer, &input_tensors](int opId, int tsId) -> TensorX {
        if (input_tensors.find({opId, tsId}) == input_tensors.end()) {
            input_tensors[{opId, tsId}] = xfer.new_tensor();
        }
        return input_tensors.at({opId, tsId});
    };

    xfer.srcOps = create_rule_graph(r.srcOp, get_input_tensor);
    xfer.dstOps = create_rule_graph(r.dstOp, get_input_tensor);
    
    for (MapOutput const &m : r.mappedOutput) {
        TensorX srcTensorX = xfer.srcOps[m.srcOpId]->outputs[m.srcTsId];
        TensorX dstTensorX = xfer.dstOps[m.dstOpId]->outputs[m.dstTsId];
        xfer.map_output(srcTensorX, dstTensorX);
    }
}

std::vector<GraphXfer*> create_xfers(FFModel *model, RuleCollection const &rules) {
    std::vector<GraphXfer*> xfers;
    for (Rule const &r : rules.rules) {
        GraphXfer *xfer = new GraphXfer(model);
        create_xfer(*xfer, r);
        xfers.push_back(xfer);
    }
    return xfers;
}

}