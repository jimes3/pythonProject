'''
#for循环
y = list(eval(input('成绩之间用英文逗号隔开：')))
for i in range(len(y)):
    if y[i] >= 60:
        print(y[i])
#列表推导式
y = list(eval(input('成绩之间用英文逗号隔开：')))
[print(y[i]) for i in range(len(y)) if y[i] >= 60]


y1 = [1,4,5,2,3,7,8,9]
y1.sort()
y1.insert(3,4)
y1.pop(3)
print(y1)
y2 = sorted(y1)
y2.insert(3,4)
y2.pop(3)
print(y2)
'''
# 创建一个学生信息字典，每个学生有多个属性
students = {
    "学生1": {
        "姓名": "张三",
        "成绩": 85,
        "年龄": 18,
        "性别": "男"
    },
    "学生2": {
        "姓名": "李四",
        "成绩": 60,
        "年龄": 19,
        "性别": "女"
    },
    "学生3": {
        "姓名": "王五",
        "成绩": 45,
        "年龄": 20,
        "性别": "男"
    }
}
# 找出不及格的学生信息
failed_students = {}
for name, information in students.items():
    if information["成绩"] < 60:
        failed_students[name] = information
# 按照课程成绩进行排序
sorted_students = dict(sorted(students.items(), key=lambda item: item[1]["成绩"]))
print("成绩不及格的学生信息：",failed_students.items())

for name, information in sorted_students.items():
    print(f"学生姓名: {information['姓名']}, 成绩: {information['成绩']}, "
          f"年龄: {information['年龄']}, 性别: {information['性别']}")

# 从键盘输入一行字符串
input_string = input("请输入一行字符串,用空格隔开: ")
# 使用空格分割字符串，得到单词列表
words = input_string.split()
# 统计单词个数
word_count = len(words)
# 输出结果
print(f"输入的字符串包含 {word_count} 个单词。")
