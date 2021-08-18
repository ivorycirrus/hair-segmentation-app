const API_SERVER_URL = "https://api.pmkanghairtest.ml/infer";
const MODEL_PATH = location.href.replace(/[^/]*$/, '') + "webmodels/model.json";

/*== Tensorflow utils ==================================================*/
async function loadModel(){
    let m = await tf.loadLayersModel(MODEL_PATH);
    console.log(new Date().toLocaleString() + " model loaded");
    return m;
};

function detectDevice(){
  console.log('mobile : ' + !!navigator.maxTouchPoints)
  return !!navigator.maxTouchPoints ? 'mobile' : 'computer'
}

var model;
var streamming = false;
var videoInput = document.getElementById('camera');
var outputRes = detectDevice()=='mobile'?{width:480, height:640}:{width:640, height:480};

function hairSegmentation(){
    return new Promise((resolve, reject) =>{
        setTimeout(()=>{
            tf.engine().startScope();

            let img = tf.browser.fromPixels(videoInput).div(255.0);
            let tensor = img.resizeNearestNeighbor([224,224]).expandDims();

            let predict = model.predict(tensor).round().squeeze(0);

            let mask = predict.mul(0.2).resizeNearestNeighbor([outputRes.height, outputRes.width]);
            let result = img.mul(0.8).add(mask);
            
            tf.browser.toPixels(result , document.getElementById('dnn-output'));

            tf.engine().endScope();

            resolve();
        },0);
    });
}

async function startProcess(){
    streamming = true;
    for(;streamming;){
        await hairSegmentation();    
    }
}

function choosePhoto(){
    streamming = false;
    $("#img_upload").click();
}

function showPhoto(imgFile){
    var reader  = new FileReader();    
    reader.readAsDataURL(imgFile);
    reader.onloadend = function (e) {
        var image = new Image();
        image.src = e.target.result;

        image.onload = function(ev) {
            var canvas = document.getElementById('selected-photo');            
            canvas.width = image.width;
            canvas.height = image.height;
            var ctx = canvas.getContext('2d');
            ctx.drawImage(image,0,0);

            $("#modal-upload").modal();
        }
    };
}

function uploadPhoto(){
    let canvas = document.getElementById('selected-photo');
    let txt64 = canvas.toDataURL();
    $.ajax({
        url: API_SERVER_URL,
        type: "post",
        accept: "application/json",
        contentType: "application/json; charset=utf-8",
        data: JSON.stringify( {image:txt64}),
        dataType: "json",
        success: function(data) {
            console.log('data received')
            document.getElementById('img-red').src = data.red;
            document.getElementById('img-green').src = data.green;
            document.getElementById('img-blue').src = data.blue;
            document.getElementById('img-yellow').src = data.yellow;
            document.getElementById('img-cyan').src = data.cyan;
            document.getElementById('img-pink').src = data.pink;

            $(".gallery").show();

            setTimeout(function(){ startProcess(); }, 3000);
        },
        error: function(jqXHR,textStatus,errorThrown) {
            alert(errorThrown)
        }
    });
}

/*== Initialize ==================================================*/
$(async () => {
    if(navigator.mediaDevices && navigator.mediaDevices.getUserMedia){
        try {
            // Start camera
            let constraints = { audio : false, video: { facingMode: 'user', width : 640 , height : 480 } };
            let stream = await navigator.mediaDevices.getUserMedia(constraints);            

            // Set camera stream
            videoInput.srcObject = stream;

            // load model
            model = await loadModel();

            // set video params
            videoInput.width = videoInput.videoWidth;
            videoInput.height = videoInput.videoHeight;

            // set output params
            let outputObj = document.getElementById('dnn-output');
            
            // image selected event handler
            $("#img_upload").change(function(e){
                if(e.target.files) showPhoto(e.target.files[0]);
            })


            // Hide loading progress
            setTimeout(()=>{ 
                startProcess();
                $("#loading").hide(); 
                console.log("end loading");
            }, 1000);
        } catch (e) {
            console.log("module init error");
            console.log(e);
            alert("카메라를 사용 할 수 없습니다.\n카메라 접근권한을 허용 해 주세요.");
        }
    }else{
        alert("이 브라우저에서는 사용 할 수 없습니다.\nChrome 또는 Safari 브라우저를 이용 해 주세요.");
    }
});