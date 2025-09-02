import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "ImageToVideoInserter.DynamicInputs",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        switch (nodeData.name) {
            case "ImageFrameSelector":
                nodeType.prototype.onNodeCreated = function () {
                    this._imageType = "IMAGE"
                    this.addWidget("button", "Update inputs", null, () => {
                        if (!this.inputs) {
                            this.inputs = [];
                        }
                        if (!this.widgets) {
                            this.widgets = [];
                        }
                        
                        const target_number_of_inputs = this.widgets.find(w => w.name === "inputcount")["value"];
                        const num_image_inputs = this.inputs.filter(input => input.type === this._imageType).length;
                        const num_frame_widgets = this.widgets.filter(w => w.name && w.name.startsWith("frame_")).length;
                        
                        if(target_number_of_inputs === num_image_inputs && target_number_of_inputs === num_frame_widgets) {
                            return; // already set, do nothing
                        }
                        
                        // Remove excess inputs and widgets
                        if(target_number_of_inputs < num_image_inputs) {
                            const inputs_to_remove = num_image_inputs - target_number_of_inputs;
                            for(let i = 0; i < inputs_to_remove; i++) {
                                this.removeInput(this.inputs.length - 1);
                            }
                        }
                        
                        if(target_number_of_inputs < num_frame_widgets) {
                            const widgets_to_remove = num_frame_widgets - target_number_of_inputs;
                            for(let i = 0; i < widgets_to_remove; i++) {
                                // 最後のframe_*ウィジェットを探して削除
                                for(let j = this.widgets.length - 1; j >= 0; j--) {
                                    if(this.widgets[j].name && this.widgets[j].name.startsWith("frame_")) {
                                        this.widgets.splice(j, 1);
                                        break;
                                    }
                                }
                            }
                        }
                        
                        // Add new inputs
                        for(let i = num_image_inputs + 1; i <= target_number_of_inputs; ++i) {
                            this.addInput(`image_${i}`, this._imageType);
                        }
                        
                        // Add new frame widgets
                        for(let i = num_frame_widgets + 1; i <= target_number_of_inputs; ++i) {
                            this.addWidget("number", `frame_${i}`, i * 10, (value) => {
                                // ウィジェット値の更新処理
                            }, {"min": 0, "max": 10000, "step": 1});
                        }
                    });
                }
                break;
        }
    }
});