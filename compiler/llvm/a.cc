#include <stdio.h>

class Animal
{
public:
    virtual void print() = 0;
};

struct Dog
{
    int age;
    char *name;
};

void PrintDog(Dog dog)
{
    printf("Dog age = %d, name = %s\n", dog.age, dog.name);
}

class Cat : public Animal
{
public:
    int age;
    char *name;

    Cat(int age, char *name)
    {
        this->age = age;
        this->name = name;
    }
    void print()
    {
        static int cnt = 0;
        cnt++;
        printf("Cat age = %d, name = %s, cnt = %d\n", age, name, cnt);
    }
};
class Lion : public Animal
{
public:
    int age;
    char *name;

    Lion(int age, char *name)
    {
        this->age = age;
        this->name = name;
    }
    void print()
    {
        printf("Lion age = %d, name = %s\n", age, name);
    }
};

void PrintAnimal(Animal *animal)
{
    animal->print();
}

int PrintAnimal(bool is_dog)
{
    int res = 0;
    if (is_dog)
    {
        res = 2;
        Dog dog{1, "dog"};
        for (int i = 0; i < 3; i++)
        {
            PrintDog(dog);
        }
    }
    else
    {
        res = 1;
        Cat *cat = new Cat(1, "cat");
        Lion *lion = new Lion(1, "lion");

        for (int i = 0; i < 3; i++)
        {
            PrintAnimal(cat);
            PrintAnimal(lion);
        }
        delete cat;
        delete lion;
    }
    return res;
}

static Dog global_dog{2, "global"};

int main()
{
    // 接收用户输入
    bool is_dog;
    printf("请输入动物类型（1: 狗, 0: 猫）: ");
    scanf("%d", &is_dog);

    // 调用打印函数
    int res = PrintAnimal(is_dog);
    printf("打印结果: %d\n", res);

    // 打印全局狗
    PrintDog(global_dog);
    return 0;
}