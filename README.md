# PPDNN
Privacy-preserving Outsourced Deep Neural Network Dual-cloud Training using Secret Sharing

訓練集: MNIST, Fashion MNIST

近年來，深度神經網絡（DNN）在人臉辨識、自然語言處理、物體檢測、物體分類等方面都取得了重大的突破。隨著資料科學的發展和資料集規模的不斷擴大，DNN訓練對計算能力和儲存空間的要求也越來越高。雖然GPU和訓練方法幾十年來一直在優化，但儲存空間的需求也是一筆龐大的費用。為了降低成本，必須將資料集集中到雲端儲存空間中，與雲端服務器進行交互式訓練。

要將資料儲存在雲端空間中，資料安全無疑是最需要考慮的問題。用於DNN訓練的資料集通常從許多不同的機構收集而來，資料內容通常包含部分隱私訊息。爲了防止隱私訊息泄露，資料再送往雲端儲存空間時必須事先加密（或通過某種方式進行保護）。

本研究主要採用秘密共享（輕量級）實現保有隱私授權的DNN訓練。使用秘密共享可以將訓練資料和模型分成兩個共享，然後授權給兩個雲端服務器來操作訓練。由於模型訓練的計算在雲端服務器上運行，模型訓練者可以節省大部分的計算量與儲存需求，因此模型訓練者也可以是計算能力較弱的設備（如：智慧穿戴裝置）的使用者。

主要貢獻：
1. 在資料收集時，Data Owners 皆傳送經過加密的 Data，因此 Trainer 或任何竊聽者皆無法從訓練資料取得任何私人訊息。
2. Trainer 可以將 DNN 訓練的計算授權給雲端伺服器執行，並確保完成訓練的 Model 不會洩漏。
3. 經過此算法得出的 DNN 模型，效能與不經過任何加密的訓輛方法得出的模型幾乎相同。

Title: Privacy-Preserving Outsourced Deep Neural Network Dual-Cloud Training using Secret Sharing

Training Sets: MNIST, Fashion MNIST

In recent years, significant breakthroughs have been achieved in deep neural networks (DNNs) across various domains such as face recognition, natural language processing, object detection, and object classification. With the development of data science and the continuous expansion of dataset sizes, the requirements for computational power and storage space in DNN training have also increased. While GPUs and training methods have been optimized over the years, storage space demands remain a substantial cost. To mitigate costs, it is essential to centralize datasets in cloud storage and engage in interactive training with cloud servers.

Storing data in cloud spaces brings forth the paramount concern of data security. Datasets used for DNN training are typically collected from various institutions and often contain sensitive information. To prevent the leakage of private information, data must be encrypted before being transmitted to cloud storage (or protected through some means).

This study primarily employs lightweight secret sharing to achieve privacy-preserving authorized DNN training. Using secret sharing, training data and models are split into two shares and authorized for operation by two cloud servers. As the computational processes of model training run on cloud servers, the model trainer can significantly reduce computational and storage requirements. Consequently, model trainers may include users with lower computational capabilities, such as those using smart wearable devices.

Key Contributions:

During data collection, data owners transmit encrypted data, ensuring that trainers or any potential eavesdroppers cannot access any private information from the training data.
Trainers can authorize the execution of DNN training computations to cloud servers, ensuring that the completed model does not leak sensitive information.
The DNN models derived from this algorithm exhibit performance almost equivalent to models obtained through training without any encryption.
