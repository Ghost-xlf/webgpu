export default class Calc{

    public a:number = 0;
    public b:number = 0;
    public readonly pi:number = 3.1415926;

    // 构造函数
    public constructor(var1:number = 0, var2:number = 0){
        this.a = var1;
        this.b = var2;
    }

    public add():number{
        this.print();
        return this.a + this.b;
    }

    public subtract():number{
        return this.a - this.b;
    }

    public multiply():number{
        return this.a * this.b;
    }

    public divide():number{

        if(!this.b){
            return 0;
        }
        else{
            return this.a / this.b;
        }
    }

    private print():void{
        console.warn(`a:${this.a} b:${this.b}`);
    }
}

abstract class Animal{

    public name:string;
    public weight:number;
    public readonly PI:number = 3.1415926;
    constructor(name:string = '', weight:number = 0){
        this.name = name;
        this.weight = weight;
    }
    public abstract helloworld():void;
}

class Human extends Animal{
    public age:number;
    constructor(name:string = '', weight:number = 0,age:number = 0){
        super(name,weight);
        this.age = age;
    }

    public helloworld():void{
        console.error("hello world");
    }
}

let zhangsan = new Human("张三", 80, 27);
zhangsan.helloworld();

let func = (var1:number, var2:number):number => {
    let temp:number = var1 + var2;
    return temp;
}
let sum = (x:number, y:number, ...items:number[]) =>{
    console.warn(items);
    return x;
}
sum(1,2,3,4,5);