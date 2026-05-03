import xml.etree.ElementTree as ET

tree = ET.parse('data/sentinel-1/S1A_AUX_INS_V20250601T000000_G20251015T084917.SAFE/S1A_AUX_INS_V20250601T000000_G20251015T084917.SAFE/data/s1a-aux-ins.xml')
root = tree.getroot()

# Print rollSteeringParams
rsp = root.find('rollSteeringParams')
print('=== rollSteeringParams ===')
for c in rsp:
    print(f'  {c.tag}: {c.text}')

# Print all swath names
print('\n=== swaths ===')
for sw in root.findall('.//swathParams'):
    name = sw.findtext('swath')
    az_rate = sw.findtext('.//azimuthSteeringRate')
    print(f'  {name}: azimuthSteeringRate={az_rate}')
    # Check for any roll or mispointing related fields
    for c in sw.iter():
        if any(k in c.tag.lower() for k in ['roll', 'miss', 'point', 'elev', 'look', 'angle', 'nominal']):
            print(f'    {c.tag}: {c.text}')

# Print remaining top-level fields
print('\n=== other top-level fields ===')
for child in root:
    if child.tag not in ('rollSteeringParams', 'swathParamsList'):
        print(f'  {child.tag}: {child.text}')
