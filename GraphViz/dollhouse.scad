// Modular Dollhouse Room Module
// Parameters
module_size = 204;
wall_thickness = 4;
floor_thickness = 5;
pin_diameter = 6;
hole_diameter = 6.2;
pin_length = 6;
rail_base = 6;
rail_tip = 4;
rail_depth = 4;
rail_angle = 60;

// Derived values
inner_size = module_size - 2*wall_thickness;
ceiling_thickness = floor_thickness;

module_room() {
    // Floor
    cube([module_size, module_size, floor_thickness]);
    translate([0, 0, floor_thickness]) {
        // Walls (hollow cube)
        difference() {
            cube([module_size, module_size, module_size - floor_thickness]);
            translate([wall_thickness, wall_thickness, 0])
                cube([inner_size, inner_size, module_size]);
        }
    }
    // Ceiling
    translate([0, 0, module_size - ceiling_thickness])
        cube([module_size, module_size, ceiling_thickness]);

    // Dovetail rails and grooves
    // Right face rail
    translate([module_size, 0, (module_size-rail_depth)/2])
        rotate([0,90,0]) dovetail_rail();
    // Left face groove
    translate([0, 0, (module_size-rail_depth)/2])
        rotate([0,90,0]) dovetail_groove();
    // Back face rail
    translate([0, module_size, (module_size-rail_depth)/2])
        rotate([90,0,0]) dovetail_rail();
    // Front face groove
    translate([0, 0, (module_size-rail_depth)/2])
        rotate([90,0,0]) dovetail_groove();

    // Dowel pins on ceiling
    for (x = [wall_thickness + pin_diameter/2, module_size - wall_thickness - pin_diameter/2])
    for (y = [wall_thickness + pin_diameter/2, module_size - wall_thickness - pin_diameter/2])
        translate([x, y, module_size - ceiling_thickness - pin_length])
            cylinder(d = pin_diameter, h = pin_length);

    // Holes on floor
    for (x = [wall_thickness + pin_diameter/2, module_size - wall_thickness - pin_diameter/2])
    for (y = [wall_thickness + pin_diameter/2, module_size - wall_thickness - pin_diameter/2])
        translate([x, y, 0])
            rotate([0,0,0])
            cylinder(d = hole_diameter, h = floor_thickness + 1);
}

module_room();

// Dovetail profile modules
module dovetail_rail() {
    pts = [ [0,0], [rail_base/2, rail_depth], [rail_tip/2, rail_depth], [rail_tip/2, 0] , [-rail_tip/2, 0], [-rail_tip/2, rail_depth], [-rail_base/2, rail_depth] ];
    linear_extrude(height = module_size)
        polygon(points = pts);
}

module dovetail_groove() {
    pts = [ [0,0], [rail_tip/2, rail_depth], [rail_base/2, rail_depth], [rail_base/2, 0] , [-rail_base/2, 0], [-rail_base/2, rail_depth], [-rail_tip/2, rail_depth] ];
    linear_extrude(height = module_size + 1) // extra to cut through
        polygon(points = pts);
}
