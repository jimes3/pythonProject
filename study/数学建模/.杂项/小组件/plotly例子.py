# 导入必要的模块
from plotly.graph_objects import Bar, Figure, layout

# 准备数据
months = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']
old_members = [1859, 3087, 3472, 2886, 2912, 2973, 3208, 3366, 3173, 2413, 2278, 2062]
new_members = [420, 1141, 1256, 755, 743, 702, 730, 709, 711, 623, 560, 559]

# 生成空图表
figure = Figure()

# 实例化老会员的柱形 trace
trace1 = Bar(x=months,
             y=old_members,
             name='老会员')

# 实例化新会员的柱形 trace
trace2 = Bar(x=months,
             y=new_members,
             name='新会员')

# 将两个 trace 添加进 figure 中
figure.add_trace(trace1)
figure.add_trace(trace2)

# 调整 trace 的宽度和颜色
figure.update_traces(width=0.4)
figure.update_traces(selector=dict(name='老会员'),
                     marker_color='CadetBlue')
figure.update_traces(selector=dict(name='新会员'),
                     marker_color='MediumAquamarine')

# 实例化一个图表标题
fig_title = layout.Title(text='健身俱乐部每月客流人数',
                         font=dict(size=20,color='CadetBlue'),
                         x=0.5)

# 将柱形 trace 排列方式和标题应用到布局中
figure.update_layout(barmode='stack',
                     title=fig_title)

figure.show()