import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "ImageToVideoInserter.DynamicInputs",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        switch (nodeData.name) {
            case "ImageFrameSelector":
                nodeType.prototype.onNodeCreated = function () {
                    this._imageType = "IMAGE"
                    
                    // 初期フレーム入力を追加
                    this.addInput("frame_1", "INT");
                    this.addInput("frame_2", "INT");
                    
                    const updateButton = this.addWidget("button", "Update inputs", null, () => {
                        if (!this.inputs) {
                            this.inputs = [];
                        }
                        if (!this.widgets) {
                            this.widgets = [];
                        }
                        
                        const target_number_of_inputs = this.widgets.find(w => w.name === "inputcount")["value"];
                        const num_image_inputs = this.inputs.filter(input => input.type === this._imageType).length;
                        const num_frame_inputs = this.inputs.filter(input => input.name && input.name.startsWith("frame_")).length;
                        
                        if(target_number_of_inputs === num_image_inputs && target_number_of_inputs === num_frame_inputs) {
                            return; // already set, do nothing
                        }
                        
                        // Remove excess inputs and widgets
                        if(target_number_of_inputs < num_image_inputs) {
                            const inputs_to_remove = num_image_inputs - target_number_of_inputs;
                            for(let i = 0; i < inputs_to_remove; i++) {
                                // 末尾の画像入力を削除（image_2, image_3, ... の順）
                                for(let j = this.inputs.length - 1; j >= 0; j--) {
                                    if(this.inputs[j].type === this._imageType) {
                                        this.removeInput(j);
                                        break;
                                    }
                                }
                            }
                        }
                        
                        if(target_number_of_inputs < num_frame_inputs) {
                            const inputs_to_remove = num_frame_inputs - target_number_of_inputs;
                            for(let i = 0; i < inputs_to_remove; i++) {
                                // 最後のframe_*入力を探して削除
                                for(let j = this.inputs.length - 1; j >= 0; j--) {
                                    if(this.inputs[j].name && this.inputs[j].name.startsWith("frame_")) {
                                        this.removeInput(j);
                                        break;
                                    }
                                }
                            }
                        }
                        
                        // Add new inputs
                        for(let i = num_image_inputs + 1; i <= target_number_of_inputs; ++i) {
                            this.addInput(`image_${i}`, this._imageType);
                        }
                        
                        // Add new frame inputs
                        for(let i = num_frame_inputs + 1; i <= target_number_of_inputs; ++i) {
                            this.addInput(`frame_${i}`, "INT");
                        }
                        
                        // Update inputsボタンを最後尾に移動
                        if (updateButton) {
                            const buttonIndex = this.widgets.indexOf(updateButton);
                            if (buttonIndex !== -1 && buttonIndex !== this.widgets.length - 1) {
                                this.widgets.splice(buttonIndex, 1);
                                this.widgets.push(updateButton);
                            }
                        }
                    });
                }
                break;
        }
    }
});