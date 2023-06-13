<template>
  <div class="header">
    <div class="title">Purify Your OWN MODEL</div>
  </div>
  <div class="wrapper">
    <div class=" content">
      <div class="content-1">
      <div class="sub-1">
        <div class="box-3">
          <h1>Original <span>Demo</span> </h1>
          <h2>Overall Rating <span>87</span></h2>
          <div class="wra-3">
            <ul>
              <li>Membership Rating <span>91</span></li>
              <li>Inversion Rating <span>81</span></li>
              <li>Attribute Rating <span>71</span></li>
            </ul>
          </div>
        </div>

      </div>
      <div style="width:3px;background-color: grey;height: 45vh;margin-top:5vh"></div>
      <div class="sub-2">
        <h1>Diagnosis Card</h1>
      </div></div>
      <div style="height:10vh"></div>
      <div class="text-new">
        <div><svg width="25" height="15" viewBox="0 0 25 15" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M0.618375 0.618373C0.90748 0.329355 1.29954 0.166994 1.70833 0.166994C2.11713 0.166994 2.50919 0.329355 2.79829 0.618373L12.5 10.3201L22.2017 0.618373C22.3439 0.471128 22.514 0.35368 22.7021 0.272883C22.8902 0.192086 23.0925 0.149557 23.2972 0.147779C23.5019 0.146 23.7049 0.185007 23.8944 0.262523C24.0839 0.340039 24.256 0.454513 24.4007 0.599264C24.5455 0.744015 24.66 0.916146 24.7375 1.10561C24.815 1.29508 24.854 1.49808 24.8522 1.70278C24.8504 1.90748 24.8079 2.10978 24.7271 2.29787C24.6463 2.48596 24.5289 2.65608 24.3816 2.79829L13.59 13.59C13.3009 13.879 12.9088 14.0413 12.5 14.0413C12.0912 14.0413 11.6991 13.879 11.41 13.59L0.618375 2.79829C0.329357 2.50918 0.166996 2.11713 0.166996 1.70833C0.166996 1.29954 0.329357 0.907478 0.618375 0.618373Z" fill="black"/>
  </svg>
  </div>
      </div>
    </div>
    <div class="content-2 content">
      <div class="wra-1">
        <h1>Prerequisites:Pytorch <span>1.12.1</span> + CUDA <span>11.3</span></h1>
        <div class="text text-1">
          <div>Input Size</div>
          <input  name="" v-model="inputSize1" >
          <div class="x">X</div>
          <input  name="" v-model="inputSize2" >
        </div>
        <div class="text text-1">
          <div>Output Size</div>
          <input  name="" v-model="outputSize1" >
          <div class="x">X</div>
          <input  name="" v-model="outputSize2" >
        </div>
        <div class="text text-2">
          <div>Option 1</div>
          <div class="opt" :class="{ 'selected': select1===1}" @click="change1(1)">xxxx</div>
          <div class="opt" :class="{ 'selected': select1===2}" @click="change1(2)">xxxx</div>
          <div class="opt" :class="{ 'selected': select1===3}" @click="change1(3)">xxxx</div>
        </div>
        <div class="text text-2">
          <div>Option 2</div>
          <div class="opt" :class="{ 'selected': select2===1}" @click="change2(1)">xxxx</div>
          <div class="opt" :class="{ 'selected': select2===2}" @click="change2(2)">xxxx</div>
          <div class="opt" :class="{ 'selected': select2===3}" @click="change2(3)">xxxx</div>
        </div>
        <div class="text text-3"></div>
      </div>
    </div>
  

  </div>
</template>

<script>
// @ is an alias to /src

export default {
  name: 'HomeView',
  components: {
  },
  data(){
    return {
      select1: 0,
      select2:0,
    }
  },
  methods:{
    change1(index){
      this.select1=index;
    },
    change2(index){
      this.select2=index;
    }
  }
}
</script>
<style>
.header {
  z-index: 100;
  position: fixed;
  width: 100%;
  height: 20vh;
  background-image: url(../assets/demo-origin.png);
  background-size: 100vw auto;
  /* 图片等比例缩放以填充容器 */
  background-position: top;
  /* 图片靠下对齐 */
  background-repeat: no-repeat;
}

.content-1{
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.sub-1{
  height:50vh;
  width:30vw;
  margin-left:10vw;
}
.sub-2{
  margin-top:5vh;
  height:45vh;
  width:40vw;
  margin-right: 10vw;
  padding: 15px 30px 15px 15px;
  border-radius: 10px;
  background: #F4F4F4;
}


.content {
  z-index: 10;
  height: 100vh;
  width: 100%;
  position: relative;
  padding-top: 20vh;
  overflow: hidden;
}
.box-3 h1{
  font-size:45px;
  font-weight: 600;
}
.box-3 span{
  color:orange;
  font-weight: 600;
}
.box-3 .wra-3{
  padding: 15px 30px 15px 15px;
  border-radius: 10px;
  background: #F4F4F4;
}
.box-3 h2{
  font-size:35px;
  margin-top:5vh;
  margin-bottom: 5vh;
}
.box-3 li{
  font-size:23px;
  color: grey;
  text-align: left;
  margin-top:10px;
}
.title {
  width: 100%;
  text-align: center;
  font-size: 40px;
  margin-top: 3vh;
  font-weight: 800;
}
@keyframes jump {
      0% {
        transform: translateY(0);
      }
      50% {
        transform: translateY(-20px);
      }
      100% {
        transform: translateY(0);
      }
    }
  .jump{
    animation: jump 1s infinite;
  }
  .text-new{
    animation: jump 1s infinite;
    width:10vw;
    margin:auto;
    margin-top:6vh;
    color:white;
    font-weight: 600;
    font-size: 25px;
  }


.wra-1{
  margin:auto;
  width:70vw;
  height: 60vh;
  padding-left:5vw;
  padding-top:2vh;
  /* background-color: #f4f4f4; */
  border-radius: 30px;
  background-color: #f4f4f4;
  -webkit-backdrop-filter: blur(10px); /* Chrome */
  backdrop-filter: blur(10px);
}
.wra-1 h1{
  text-align: left;
  font-weight: 800;
}
.wra-1 h1 span{
  color:orange;
}
.text{
  height: 8vh;
}
.text div:first-child{
  width:10vw !important;
  height: 3vh;
  line-height: 3vh;
  text-align: left;
  font-weight: 600;
}
.text div:first-child:before {
  content: "";
  display: inline-block;
  vertical-align: middle;
  width: 2px;
  height:100%;
  background-color: rgb(185, 185, 185);
  margin-right: 10px;
}

.text-1{
  display: flex;
  justify-content: flex-start;
  align-items: center;
}
.text-1 input{
  width:7vw;
  height: 3vh;
  line-height: 3vh;
  margin-right: 2vw;
  border-radius: 10px;
  border:2px solid rgb(185, 185, 185);
  outline: none;
}
.x{
  margin-right: 2vw;
}
.text-2 {
  display: flex;
  justify-content: flex-start;
  align-items: center;
}
.opt{
  width:10vw;
  height: 3.5vh;
  line-height: 3.5vh;
  border-radius: 3px;
  background-color: white;
  margin-right: 2vw;
  border:2px solid rgb(185, 185, 185);
}
.selected{
  color:white !important;
  background-color: orange !important;
}

</style>
