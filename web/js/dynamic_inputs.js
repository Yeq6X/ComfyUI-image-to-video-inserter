import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "ImageToVideoInserter.DynamicInputs",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        switch (nodeData.name) {
            case "ImageBatchAssembler":
            case "ImageFrameSelector": // backward compatibility
                nodeType.prototype.onNodeCreated = function () {
                    this._imageType = "IMAGE"

                    const updateInputs = () => {
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
                    };

                    const updateButton = this.addWidget("button", "Update inputs", null, updateInputs);

                    // ワークフロー読み込み時の初期化
                    setTimeout(() => {
                        updateInputs();
                        // ノードサイズを強制更新
                        this.setDirtyCanvas(true, true);
                        this.setSize?.(this.computeSize());
                        // 追加の更新処理
                        if (this.graph && this.graph.canvas) {
                            this.graph.canvas.setDirty(true, true);
                        }
                    }, 100);
                }

                // ワークフロー読み込み後の処理も追加
                const originalConfigure = nodeType.prototype.configure;
                nodeType.prototype.configure = function(info) {
                    originalConfigure?.call(this, info);
                    // configure後に入力数を調整
                    setTimeout(() => {
                        if (this.widgets) {
                            const inputcountWidget = this.widgets.find(w => w.name === "inputcount");
                            if (inputcountWidget) {
                                const target_number_of_inputs = inputcountWidget.value;
                                const num_image_inputs = this.inputs ? this.inputs.filter(input => input.type === this._imageType).length : 0;

                                if(target_number_of_inputs !== num_image_inputs) {
                                    // Remove excess inputs
                                    if(target_number_of_inputs < num_image_inputs) {
                                        const inputs_to_remove = num_image_inputs - target_number_of_inputs;
                                        for(let i = 0; i < inputs_to_remove; i++) {
                                            for(let j = this.inputs.length - 1; j >= 0; j--) {
                                                if(this.inputs[j].type === this._imageType) {
                                                    this.removeInput(j);
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                    // Add new inputs
                                    else {
                                        for(let i = num_image_inputs + 1; i <= target_number_of_inputs; ++i) {
                                            this.addInput(`image_${i}`, this._imageType);
                                        }
                                    }
                                    // ノードサイズを強制更新
                                    this.setDirtyCanvas(true, true);
                                    this.setSize?.(this.computeSize());
                                    // 追加の更新処理
                                    if (this.graph && this.graph.canvas) {
                                        this.graph.canvas.setDirty(true, true);
                                    }
                                }
                            }
                        }
                    }, 50);
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
