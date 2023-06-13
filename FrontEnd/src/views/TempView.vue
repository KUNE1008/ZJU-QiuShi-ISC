<template>
    <div class="carousel-container" id="container" :style="{ transform: `translateX(${translateX}px)` }">
      <div class="carousel-slide slide-1" v-show="currentIndex===0">
  
      </div>
      <div class="carousel-slide slide-2" v-show="currentIndex===1">
  
      </div>
      <div class="carousel-slide slide-3" v-show="currentIndex===2">
  
      </div>
    </div>
    <button class="carousel-btn prev-btn" @click="Prev" >Previous</button>
    <button class="carousel-btn next-btn" @click="Next">Next</button>
  </template>
  
  <script>
  export default {
    data() {
      return {
        currentIndex: 0,
        translateX: 0,
        slideWidth: 100,
      };
    },
    methods: {
      Prev() {
        this.translateX = -this.slideWidth;
        console.log(this.translateX);
        this.currentIndex = (this.currentIndex - 1 + 3) % 3;
      },
      Next() {
        
        this.translateX = -this.slideWidth;
        setTimeout(() => {
          // 在延迟时间之后执行的代码
          this.currentIndex = (this.currentIndex + 1) % 3;
          const con = document.getElementById('container');
        con.style.transition = 'none';
        this.translateX = 0;
        setTimeout(()=>{con.style.transition = 'transform 0.5s ease-in-out';},200);
        
          
          
          
        }, 500); 
        
        
      },
      
    },
    mounted(){
      var con=document.getElementById('container');
      this.slideWidth = con.offsetWidth;
      console.log(this.slideWidth);
    }
  };
  </script>
  
  <style>
  .slide-1 {
    width: 100%;
    height: 80vh;
    background-color: rgb(130, 130, 213);
  }
  
  .slide-2 {
    width: 100%;
    height: 80vh;
    background-color: rgb(121, 215, 170);
  }
  
  .slide-3 {
    width: 100%;
    height: 80vh;
    background-color: rgb(202, 182, 120);
  }
  
  .carousel-container {
    position: relative;
    width: 100%;
    height: 400px;
    /* 调整轮播图的高度 */
    overflow: hidden;
    display: flex;
    transition: transform 0.5s ease-in-out;
    overflow: hidden;
  }
  
  .carousel-slide {
  
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    /* display: none; */
    /* transition: opacity 0.5s ease-in-out; */
    /* transition: transform 0.5s ease-in-out; */
    flex: 0 0 100%;
  }
  
  .carousel-slide.active {
    display: block;
  }
  
  
  .carousel-btn {
    position: absolute;
    top: 50%;
    transform: translateY(-50%);
    padding: 10px;
    background-color: #333;
    color: #fff;
    border: none;
    cursor: pointer;
  }
  
  .prev-btn {
    left: 10px;
  }
  
  .next-btn {
    right: 10px;
  }
  
  </style>
  