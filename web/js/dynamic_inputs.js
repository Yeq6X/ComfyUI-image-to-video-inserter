import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "ImageToVideoInserter.DynamicInputs",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        switch (nodeData.name) {
            case "ImageBatchAssembler":
            case "ImageFrameSelector": // backward compatibility
                nodeType.prototype.onNodeCreated = function () {
                    this._imageType = "IMAGE"
                    
                    const updateButton = this.addWidget("button", "Update inputs", null, () => {
                        if (!this.inputs) {
                            this.inputs = [];
                        }
                        if (!this.widgets) {
                            this.widgets = [];
                        }
                        
                        const target_number_of_inputs = this.widgets.find(w => w.name === "inputcount")["value"];
                        const num_image_inputs = this.inputs.filter(input => input.type === this._imageType).length;
                        
                        if(target_number_of_inputs === num_image_inputs) {
                            return; // already set, do nothing
                        }
                        
                        // Remove excess inputs
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
                        
                        // Add new inputs
                        for(let i = num_image_inputs + 1; i <= target_number_of_inputs; ++i) {
                            this.addInput(`image_${i}`, this._imageType);
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
            case "WanVideoLatentInsertFrames":
                nodeType.prototype.onNodeCreated = function () {
                    this._type = "LATENT"
                    this.addWidget("button", "Update inputs", null, () => {
                        if (!this.inputs) {
                            this.inputs = [];
                        }
                        const target_number_of_inputs = this.widgets.find(w => w.name === "inputcount")["value"];
                        const num_inputs = this.inputs.filter(input => input.type === this._type).length - 1; // -1 for the samples input
                        if(target_number_of_inputs === num_inputs) return; // already set, do nothing

                        if(target_number_of_inputs < num_inputs) {
                            const inputs_to_remove = num_inputs - target_number_of_inputs;
                            for(let i = 0; i < inputs_to_remove; i++) {
                                this.removeInput(this.inputs.length - 1);
                            }
                        }
                        else {
                            for(let i = num_inputs + 1; i <= target_number_of_inputs; ++i)
                                this.addInput(`latent_${i}`, this._type)
                        }
                    });
                }
                break;
        }
    }
});
