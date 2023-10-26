def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    # 将列表分成更小的部分
    mid = len(arr) // 2
    left_half = arr[:mid]
    right_half = arr[mid:]

    # 递归调用每个半部分
    left_half = merge_sort(left_half)
    right_half = merge_sort(right_half)

    # 合并排序后的半部分
    return merge(left_half, right_half)


def merge(left, right):
    result = []
    left_index, right_index = 0, 0

    # 比较左右两部分的元素，并按顺序添加到结果列表中
    while left_index < len(left) and right_index < len(right):
        if left[left_index] < right[right_index]:
            result.append(left[left_index])
            left_index += 1
        else:
            result.append(right[right_index])
            right_index += 1

    # 将剩余的元素添加到结果列表中
    if left_index < len(left):
        result.extend(left[left_index:])
    if right_index < len(right):
        result.extend(right[right_index:])

    return result


# 测试归并排序
arr = [38, 27, 43, 3, 9, 82, 10]
sorted_arr = merge_sort(arr)
print("Sorted array:", sorted_arr)
