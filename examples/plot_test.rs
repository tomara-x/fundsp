use fundsp::hacker32::*;
use fundsp::maps;
use plotters::prelude::*;

fn seg() -> An<Segment> {
    An(Segment::new())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gate = An(Unsteady::new(vec![0.2, 0.5], true)) >> feedback(maps::rem2());
    let g = (gate | dc(0.2) | dc(0.1) | dc(0.4) | dc(0.2)) >> An(Adsr::new(false, false, true));
    //let trig = An(Unsteady::new(vec![0.1, 0.5], true));
    //let h = (pass() | dc(0.6) | dc(1.) | dc(1.) | dc(0.)) >> seg() >> (pass() | sink());
    //let f = (pass() | dc(0.1) | dc(1.) | dc(1.) | dc(1.)) >> seg() >> (pass() + h);
    //let g = (trig | dc(0.3) | dc(1.) | dc(0.) | dc(1.)) >> seg() >> (pass() | f) >> (pass() + pass());
    //let mut g = Kr::new(Box::new(g), 1024, true);
    let mut g = g;
    //let mut g = g >> ((dc(0.001) | dc(1.) | pass()) >> fundsp::maps::xerp());
    g.set_sample_rate(44100.);

    let root = BitMapBackend::new("plot.png", (1280, 640)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0f32..4f32, -2f32..2f32)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(
        LineSeries::new(
            (0..=44100 * 4)
                .map(|x| x as f32 / 44100.0)
                .map(|x| (x, g.get_mono())),
            RGBColor(0, 64, 192).stroke_width(2),
        )
        .point_size(1),
    )?;
    Ok(())
}
