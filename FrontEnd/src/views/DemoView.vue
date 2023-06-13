<template>
  <div class="header">
    <div class="title">Test on our provided DEMO</div>
  </div>
  <div class="wrapper">
    <div class="content-1 content">
      <h1>What is <span>Data Inference Attack</span>?</h1>
      <ul>
        <li>Membership Inference Attack </li>
        <li>Model Inversion Attack: </li>
        <li>Attribute Inference Attack: </li>
      </ul>
      <div class="wra">
        <div class="btn">Generate DEMO</div>
        <div style="width:10vw;"></div>
        <div class="box-1">
          <table>
    <tr v-for="(row, rowIndex) in tableData" :key="rowIndex">
      <td v-for="(cell, colIndex) in row" :key="colIndex" :class="{ 'selected': selectedIndex[rowIndex] === colIndex }" @click="selectCell(rowIndex, colIndex)">{{ cell }}</td>
    </tr>
  </table>
        </div>
      </div>
      <div class="text-new">
        <div><svg width="25" height="15" viewBox="0 0 25 15" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M0.618375 0.618373C0.90748 0.329355 1.29954 0.166994 1.70833 0.166994C2.11713 0.166994 2.50919 0.329355 2.79829 0.618373L12.5 10.3201L22.2017 0.618373C22.3439 0.471128 22.514 0.35368 22.7021 0.272883C22.8902 0.192086 23.0925 0.149557 23.2972 0.147779C23.5019 0.146 23.7049 0.185007 23.8944 0.262523C24.0839 0.340039 24.256 0.454513 24.4007 0.599264C24.5455 0.744015 24.66 0.916146 24.7375 1.10561C24.815 1.29508 24.854 1.49808 24.8522 1.70278C24.8504 1.90748 24.8079 2.10978 24.7271 2.29787C24.6463 2.48596 24.5289 2.65608 24.3816 2.79829L13.59 13.59C13.3009 13.879 12.9088 14.0413 12.5 14.0413C12.0912 14.0413 11.6991 13.879 11.41 13.59L0.618375 2.79829C0.329357 2.50918 0.166996 2.11713 0.166996 1.70833C0.166996 1.29954 0.329357 0.907478 0.618375 0.618373Z" fill="black"/>
  </svg>
  </div>
      </div>
    </div>
    
    <div class="content-2 content">
      <div class="wra-2">
        <div class="btn-wra">
          <div @click="showCard1" class="btn-2" :class="{ activeO: currentCard === 'card1' }">Evaluate</div>
          <div @click="showCard2" class="btn-2" :class="{ activeB: currentCard === 'card2' }">Purify</div>
        </div>
        <div class="card-wra">
          <div class="card" v-if="currentCard === 'card1'" :class="{'card-left': currentCard === 'card1', 'card-right': currentCard === 'card2'}">
            <!-- Card 1 的内容 -->
            <h2>Card 1</h2>
            <p>111111.</p>
          </div>
          <div class="card" v-else :class="{'card-left': currentCard === 'card2', 'card-right': currentCard === 'card1'}">
            <!-- Card 2 的内容 -->
            <h2>Card 2</h2>
            <p>222222.</p>
      </div>
        </div>
      </div>
      <div class="text-new">
        <div><svg width="25" height="15" viewBox="0 0 25 15" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M0.618375 0.618373C0.90748 0.329355 1.29954 0.166994 1.70833 0.166994C2.11713 0.166994 2.50919 0.329355 2.79829 0.618373L12.5 10.3201L22.2017 0.618373C22.3439 0.471128 22.514 0.35368 22.7021 0.272883C22.8902 0.192086 23.0925 0.149557 23.2972 0.147779C23.5019 0.146 23.7049 0.185007 23.8944 0.262523C24.0839 0.340039 24.256 0.454513 24.4007 0.599264C24.5455 0.744015 24.66 0.916146 24.7375 1.10561C24.815 1.29508 24.854 1.49808 24.8522 1.70278C24.8504 1.90748 24.8079 2.10978 24.7271 2.29787C24.6463 2.48596 24.5289 2.65608 24.3816 2.79829L13.59 13.59C13.3009 13.879 12.9088 14.0413 12.5 14.0413C12.0912 14.0413 11.6991 13.879 11.41 13.59L0.618375 2.79829C0.329357 2.50918 0.166996 2.11713 0.166996 1.70833C0.166996 1.29954 0.329357 0.907478 0.618375 0.618373Z" fill="black"/>
  </svg>
  </div>
      </div></div>

    <div class="content-3 content">
      <div class="content-3-wra">
        <div class="box-3">
        <h1>Original <span>Demo</span> </h1>
        <h2>Overall Rating <span>81</span></h2>
        <div class="wra-3">
          <ul>
            <li>Membership Rating <span>83</span></li>
            <li>Inversion Rating <span>81</span></li>
            <li>Attribute Rating <span>81</span></li>
          </ul>
        </div>
      </div>
      <div class="tu">
        <div style="height: 15vh;"></div>
        <svg width="79" height="67" viewBox="0 0 79 67" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M27.0002 49L48.0002 49V63.7C48.0002 64.9 48.9002 66.1 50.1002 66.4C51.3002 66.7 52.5002 66.4 53.4002 65.5L77.4002 35.8C78.3002 34.6 78.3002 33.1 77.4002 31.9L53.4002 1.90001C52.8002 1.30001 51.9002 0.700012 51.0002 0.700012C50.7002 0.700012 50.4002 0.700012 50.1002 1.00002C48.9002 1.30001 48.0002 2.50002 48.0002 3.70001V18.7H27.0002C25.2002 18.7 24.0002 19.9 24.0002 21.7L24.0002 45.7C24.0002 47.8 25.2002 49 27.0002 49ZM3.0002 49C4.80019 49 6.0002 47.8 6.0002 46V22C6.0002 20.2 4.80019 19 3.0002 19C1.2002 19 0.000198364 20.2 0.000198364 22V46C0.000198364 47.8 1.2002 49 3.0002 49ZM15.0002 49C16.8002 49 18.0002 47.8 18.0002 46L18.0002 22C18.0002 20.2 16.8002 19 15.0002 19C13.2002 19 12.0002 20.2 12.0002 22V46C12.0002 47.8 13.2002 49 15.0002 49Z" fill="#8A8A8A"/>
</svg>
<div style="margin-top:8vh" @click="changeToPurify">
<div class="jump"><svg width="62" height="62" viewBox="0 0 62 62" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M61.2329 8.14329C54.5272 3.06122 44.2081 0.0270996 33.6294 0.0270996C20.5435 0.0270996 9.88143 4.59572 4.37506 12.5608C1.7885 16.3021 0.358622 20.7312 0.124184 25.728C-0.0850656 30.1765 0.662809 35.0958 2.34456 40.3891C8.08925 23.1705 24.1298 9.68554 42.6252 9.68554C42.6252 9.68554 25.3194 14.2406 14.4384 28.3456C14.4326 28.3533 14.2873 28.5335 14.0374 28.8726C11.8519 31.7963 9.94731 35.1191 8.52325 38.8817C6.10912 44.6225 3.87325 52.5023 3.87325 61.998H11.6232C11.6232 61.998 10.4472 54.5987 12.4932 46.0873C15.878 46.5445 18.9024 46.7712 21.6266 46.7712C28.7507 46.7712 33.8192 45.229 37.5741 41.9197C40.9376 38.9534 42.7937 34.9699 44.7564 30.752C47.7557 24.3098 51.1541 17.0073 61.0217 11.3673C61.5875 11.0437 61.9517 10.4605 61.9944 9.80954C62.037 9.15854 61.7502 8.53272 61.231 8.13941L61.2329 8.14329Z" fill="#25B822"/>
</svg></div>
<div>
  Purify your own model
</div>

</div>

      </div>
      <div class="box-3 box-3-new">
        <h1>Purified <span>Demo</span> </h1>
        <h2>Overall Rating <span>97</span><sup>19.75%<svg width="19" height="23" viewBox="0 0 19 23" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M10.9375 5.27131V23H8.0625V5.27131L2.79119 10.5412L0.875 8.625L9.5 0L18.125 8.625L16.2088 10.5412L10.9375 5.27131Z" fill="#D81E06"/>
</svg></sup>
</h2>
        <div class="wra-3">
          <ul>
            <li>Membership Rating <span>92 </span> <svg width="10" height="13" viewBox="0 0 19 23" fill="none" >
<path d="M10.9375 5.27131V23H8.0625V5.27131L2.79119 10.5412L0.875 8.625L9.5 0L18.125 8.625L16.2088 10.5412L10.9375 5.27131Z" fill="#D81E06"/>
</svg>
</li>
            <li>Inversion Rating <span>94 </span> <svg width="10" height="13" viewBox="0 0 19 23" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M10.9375 5.27131V23H8.0625V5.27131L2.79119 10.5412L0.875 8.625L9.5 0L18.125 8.625L16.2088 10.5412L10.9375 5.27131Z" fill="#D81E06"/>
</svg>
</li>
            <li>Attribute Rating <span>91 </span> <svg width="10" height="13" viewBox="0 0 19 23" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M10.9375 5.27131V23H8.0625V5.27131L2.79119 10.5412L0.875 8.625L9.5 0L18.125 8.625L16.2088 10.5412L10.9375 5.27131Z" fill="#D81E06"/>
</svg>
</li>
          </ul>
        </div>
      </div>
      </div>
      

    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      index: 0,
      flag: 0,
      last: 0,
      tableData: [
        ['Dataset', 'B', 'C', 'D'],
        ['Backbone', 'F', 'G', 'H'],
        ['Layer', 'J', 'K', 'L'],
        ['Epoch', 'N', 'O', 'P']
      ],
      selectedIndex: [1,1,1,1],
      currentCard: 'card1'
    };
  },
  methods: {
    nextSide() {
      this.index++;
      this.index = this.index % 3;
      const contentElements = document.getElementsByClassName('content');
      console.log(contentElements[this.index].offsetTop);
      // window.removeEventListener('scroll', this.autoScroll);
      window.scrollTo({
        top: contentElements[this.index].offsetTop,
        behavior: 'smooth'
      });
      // setTimeout(() => {
      //   window.addEventListener('scroll', this.autoScroll);
      // }, 1000); 
    },
    preSlide() {
      this.index = this.index - 1 + 3;
      this.index = this.index % 3;
      const contentElements = document.getElementsByClassName('content');
      console.log(contentElements[this.index].offsetTop);
      // window.removeEventListener('scroll', this.autoScroll);
      window.scrollTo({
        top: contentElements[this.index].offsetTop,
        behavior: 'smooth'
      });
      // setTimeout(() => {
      // window.addEventListener('scroll', this.autoScroll);
      // }, 1000); 
    },
    autoScroll() {
      var now = window.pageYOffset || document.documentElement.scrollTop;
      if (this.flag != 0) {
        this.last = now;
        return;
      }
      var temp = now - this.last;
      if (temp < 0 && this.lag == 0) {
        this.flag = 1;
        this.nextSide();
        setTimeout(() => {
          this.flag = 0;
        }, 1000);
      }
      else if (temp > 0 && this.flag == 0) {
        this.flag = 1;
        this.preSlide();
        setTimeout(() => {
          this.flag = 0;
        }, 1000);
      }
      this.last = now;
    },
    selectCell(rowIndex, colIndex) {
      this.selectedIndex[rowIndex] = colIndex; // 更新选中的元素的列索引
    },
    showCard1() {
      this.currentCard = 'card1';
    },
    showCard2() {
      this.currentCard = 'card2';
    }
  },
  mounted() {
    // const contentElements = document.getElementsByClassName('content');
    // window.onscroll = function() {
    //   window.scrollTo({
    //       top: contentElements[this.index].offsetTop,
    //     });
    // };
    // this.last=window.pageYOffset || document.documentElement.scrollTop;
    // window.addEventListener('scroll', this.autoScroll);
  }
};
</script>

<style>
.wrapper {
  position: relative;
  width: 100%;
  overflow: hidden;

}

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

.title {
  width: 100%;
  text-align: center;
  font-size: 40px;
  margin-top: 3vh;
  font-weight: 800;
}

/* content 1 */
.content {
  z-index: 10;
  height: 100vh;
  width: 100%;
  position: relative;
  padding-top: 20vh;
  overflow: hidden;
}

.content-1 {
  text-align: left;
  padding-left: 4vw;
}

.content-1 h1 {
  margin-left: 40px;
}

.content-1 h1 span {
  color: rgb(255, 140, 0);
  font-size: 35px;
}

.content-1 li {
  font-size: 20px;
  font-weight: 900;
  margin-top: 2vh;
}

.wra {
  width: 60vw;
  position: relative;
  display: flex;
  justify-content: space-between;
  
  align-items: center;
  margin:auto;
  margin-top: 5vh;
  /* padding-left:10vw;
  padding-right:10vw; */
}

.btn {
  display: block;
  /* position: absolute; */
  border-radius: 1000px;
  background-color: rgb(57, 161, 236);
  color: white;
  font-size: 20px;
  font-weight: 600;
  width: 200px;
  height: 60px;
  line-height: 60px;
  text-align: center;
}

.wra .box-1 {
  font-size: 20px;
  border-radius: 20px;
  background-color: rgba(236, 236, 236, 0.973);
  padding: 10px;
  backdrop-filter: blur(15px);
}
.box-1 tr td{
  width:130px;
  /* margin:10px; */
  
  height:50px;
  text-align: center;
}
.box-1 table{
  border-spacing: 15px;
}
.box-1 tr td:first-child {
  color:black;
  font-weight: 600;
}
tr td:nth-child(n+2):nth-child(-n+4) {
  background-color: white;
}

.selected {
  background-color: orange !important;
  color: white;
}


/* content -2 */
.wra-2{
  width:80vw;
  margin:auto;
  position: relative;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding:30px;
  background: #F4F4F4;
/* filter: blur(10px); */
  border-radius: 30px;
}
.btn-wra{
  display: flex;
  flex-direction: column;
  align-items: center;
}
.card-wra{
  
  height:50vh;
  width:69vw;
  border: solid white 4px;
  transition: 0.5s ease-in-out;
}
.card{
  transition:  0.5s ease-in-out;
}
.btn-2{
  width:240px;
  font-size:30px;
  font-weight: 600;
  color:grey;
  transition:  0.3s ease-in-out;
  position: relative;
}
.btn-2:hover::before{
  content: '> ';
  position: absolute;
    left: 0;
    top: 0;
    margin-right: 5px;
}
.activeO{
  color:orange;
  font-size:40px;
}
.activeO::before{
  content:"> ";
  position: absolute;
    left: 0;
    top: 0;
    margin-right: 5px;
}
.activeB{
  color:rgb(0, 157, 255);
  font-size:40px;
}
.activeB::before{
  content:"> ";
  position: absolute;
    left: 0;
    top: 0;
    margin-right: 5px;
}
.card-left {
  /* transform: translateX(-100%); */
}
.card-right {
  /* transform: translateX(100%); */
}

.content-3-wra {
  margin:auto;
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 70vw;
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
.box-3-new span{
  color: rgb(26, 167, 26);
}
.box-3-new sup{
  font-size:20px;
  color:rgb(210, 4, 4);
}
.box-3-new svg{
  width:15px !important;
  height: 20px !important;
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
</style>
