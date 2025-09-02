import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "ImageToVideoInserter.DynamicInputs",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        switch (nodeData.name) {
            case "ImageFrameSelector":
                nodeType.prototype.onNodeCreated = function () {
                    this._imageType = "IMAGE"
                    this._frameType = "INT"
                    this.addWidget("button", "Update inputs", null, () => {
                        if (!this.inputs) {
                            this.inputs = [];
                        }
                        const target_number_of_inputs = this.widgets.find(w => w.name === "inputcount")["value"];
                        const num_image_inputs = this.inputs.filter(input => input.type === this._imageType).length - 2; // -2 for the default image_1, image_2
                        
                        if(target_number_of_inputs === num_image_inputs + 2) return; // already set, do nothing
                        
                        // Remove excess inputs
                        if(target_number_of_inputs < num_image_inputs + 2) {
                            const inputs_to_remove = (num_image_inputs + 2) - target_number_of_inputs;
                            for(let i = 0; i < inputs_to_remove * 2; i++) { // *2 because we remove both image and frame inputs
                                this.removeInput(this.inputs.length - 1);
                            }
                        }
                        // Add new inputs
                        else {
                            for(let i = num_image_inputs + 3; i <= target_number_of_inputs; ++i) {
                                this.addInput(`image_${i}`, this._imageType);
                                this.addInput(`frame_${i}`, this._frameType);
                            }
                        }
                    });
                }
                break;
        }
    }
});