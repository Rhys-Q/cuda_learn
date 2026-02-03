#include <stdio.h>

class Animal
{
    virtual void print() = 0;
};

struct Dog
{
    int age;
    char *name;
};

void PrintDog(Dog dog)
{
    printf("age = %d, name = %s\n", dog.age, dog.name);
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
        printf("age = %d, name = %s, cnt = %d\n", age, name, cnt);
    }
};

int PrintAnimal(bool is_cat)
{
    int res = 0;
    if (is_cat)
    {
        res = 1;
        Cat *cat = new Cat(1, "cat");
        for (int i = 0; i < 3; i++)
        {
            cat->print();
        }
        delete cat;
    }
    else
    {
        res = 2;
        Dog dog{1, "dog"};
        for (int i = 0; i < 3; i++)
        {
            PrintDog(dog);
        }
    }
    return res;
}

static Dog global_dog{2, "global"};

int main()
{
    // 接收用户输入
    bool is_cat;
    printf("请输入动物类型（1: 猫, 0: 狗）: ");
    scanf("%d", &is_cat);

    // 调用打印函数
    int res = PrintAnimal(is_cat);
    printf("打印结果: %d\n", res);

    // 打印全局狗
    PrintDog(global_dog);
    return 0;
}