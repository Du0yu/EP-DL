from eppy.modeleditor import IDF
# 初始化 Eppy 和加载 IDF 文件
idf_path = "5Zone.idf"  # 替换为您的 IDF 文件路径
idd_file_path = r"C:\EnergyPlusV24-1-0\Energy+.idd"
IDF.setiddname(idd_file_path)
idf1 = IDF(idf_path)

print(idf1.idfobjects)
#
# 获取建筑对象
buildings = idf1.idfobjects['Building']
print("Buildings:")
for building in buildings:
    print(" -", building.Name)

# 获取区域对象
zones = idf1.idfobjects['Zone']
print("\nZones:")
for zone in zones:
    print(" -", zone.Name)

# 获取区域 HVAC 设备列表对象
hvac_equipment_lists = idf1.idfobjects['ZoneHVAC:EquipmentList']
print("\nHVAC Equipment Lists:")
for equipment_list in hvac_equipment_lists:
    print(" -", equipment_list.Name)

# 获取区域 HVAC 模板对象
hvac_templates = idf1.idfobjects['HVACTemplate:Zone']
print("\nHVAC Templates:")
for template in hvac_templates:
    print(" -", template.Name)

# 获取风扇对象
fans = idf1.idfobjects['Fan:ConstantVolume']
print("\nConstant Volume Fans:")
for fan in fans:
    print(" -", fan.Name)

# 获取线圈对象
cooling_coils = idf1.idfobjects['Coil:Cooling:Water']
print("\nCooling Coils:")
for coil in cooling_coils:
    print(" -", coil.Name)

heating_coils = idf1.idfobjects['Coil:Heating:Water']
print("\nHeating Coils:")
for coil in heating_coils:
    print(" -", coil.Name)

# 获取照明对象
lights = idf1.idfobjects['Lights']
print("\nLights:")
for light in lights:
    print(" -", light.Name)

# 获取室内照明对象
interior_lights = idf1.idfobjects['InteriorLights']
print("\nInterior Lights:")
for interior_light in interior_lights:
    print(" -", interior_light.Name)

# 获取建筑材料对象
materials = idf1.idfobjects['Material']
print("\nMaterials:")
for material in materials:
    print(" -", material.Name)

# 获取构造对象
constructions = idf1.idfobjects['Construction']
print("\nConstructions:")
for construction in constructions:
    print(" -", construction.Name)

# 获取建筑表面对象
surfaces = idf1.idfobjects['BuildingSurface:Detailed']
print("\nBuilding Surfaces:")
for surface in surfaces:
    print(" -", surface.Name)

# 获取建筑电力消耗对象
electricity_building = idf1.idfobjects['Electricity:Building']
print("\nElectricity Building:")
for electricity in electricity_building:
    print(" -", electricity.Name)

# 获取设备电力消耗对象
electricity_equipment = idf1.idfobjects['Electricity:Equipment']
print("\nElectricity Equipment:")
for equipment in electricity_equipment:
    print(" -", equipment.Name)

# 获取负荷调度对象
load_schedules = idf1.idfobjects['LoadSchedule']
print("\nLoad Schedules:")
for schedule in load_schedules:
    print(" -", schedule.Name)

# 获取恒定调度对象
constant_schedules = idf1.idfobjects['Schedule:Constant']
print("\nConstant Schedules:")
for schedule in constant_schedules:
    print(" -", schedule.Name)