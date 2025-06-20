import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.io.*;

public class NozzleSimulator {

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            System.out.println("Diretório atual: " + System.getProperty("user.dir"));

            JFrame frame = new JFrame("Porto Space Team Nozzle Geometry Simulator");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setExtendedState(JFrame.MAXIMIZED_BOTH);
            frame.setLayout(new BorderLayout());

            // Center Panel
            JPanel centerPanel = new JPanel(new BorderLayout());
            centerPanel.setBackground(Color.WHITE);

            try {
                // Simple Paht to the Image
                String imagePath = "PST.png";

                File imageFile = new File(imagePath);
                System.out.println("Image Exists? " + imageFile.exists());
                System.out.println("Absolute Path: " + imageFile.getAbsolutePath());

                BufferedImage bufferedImage = ImageIO.read(imageFile);
                if (bufferedImage == null) {
                    throw new IOException("Image Format Not Supported or Corrupted File!");
                }

                JLabel logoLabel = new JLabel(new ImageIcon(bufferedImage));
                logoLabel.setHorizontalAlignment(JLabel.CENTER);
                centerPanel.add(logoLabel, BorderLayout.CENTER);
            } catch (Exception ex) {
                JLabel logoLabel = new JLabel("Image PST.png Not Found!");
                logoLabel.setForeground(Color.RED);
                logoLabel.setFont(new Font("Arial", Font.BOLD, 24));
                logoLabel.setHorizontalAlignment(JLabel.CENTER);
                centerPanel.add(logoLabel, BorderLayout.CENTER);
                ex.printStackTrace();
            }

            JLabel title = new JLabel("Porto Space Team Nozzle Geometry Simulator");
            title.setFont(new Font("Roman", Font.BOLD, 40));
            title.setHorizontalAlignment(JLabel.CENTER);
            title.setBorder(BorderFactory.createEmptyBorder(150, 0, 150, 0));
            centerPanel.add(title, BorderLayout.SOUTH);

            frame.add(centerPanel, BorderLayout.CENTER);

            // Side Panel
            JPanel sidePanel = new JPanel();
            sidePanel.setLayout(new BoxLayout(sidePanel, BoxLayout.Y_AXIS));
            sidePanel.setBackground(new Color(30, 30, 30));
            sidePanel.setPreferredSize(new Dimension(300, 0));

            JButton runButton = new JButton("Execute Simulation");
            JButton loadParamsButton = new JButton("Load Parameters");
            JButton exitButton = new JButton("Exit");
            JButton throatButton = new JButton("Calculate Throat Radius");
            JButton analysisButton = new JButton("Expansion Ratio Analysis");
            JButton thrustButton = new JButton("Plot Thrust");
            JButton performanceButton = new JButton("Performance Parameters");

            Font buttonFont = new Font("Arial", Font.PLAIN, 18);
            for (JButton b : new JButton[]{runButton, loadParamsButton, throatButton, analysisButton, thrustButton, performanceButton,
             exitButton}) {
                b.setFont(buttonFont);
                b.setAlignmentX(Component.CENTER_ALIGNMENT);
                b.setMaximumSize(new Dimension(250, 50));
                b.setFocusPainted(false);
                b.setBackground(new Color(220, 220, 220));
                b.setMargin(new Insets(10, 20, 10, 20));
                if (b != exitButton) {
                    sidePanel.add(Box.createVerticalStrut(30));
                    sidePanel.add(b);
                }         
            }

            // Personalizing Exhaust Plot Button
            JButton exhaustButton = new JButton("<html><center>Plot Exhaust Velocity<br>and Exhaust Temperature</center></html>");
            exhaustButton.setFont(new Font("Arial", Font.PLAIN, 18));
            exhaustButton.setAlignmentX(Component.CENTER_ALIGNMENT);
            exhaustButton.setMaximumSize(new Dimension(250, 80));
            exhaustButton.setFocusPainted(false);
            exhaustButton.setBackground(new Color(220, 220, 220));
            exhaustButton.setMargin(new Insets(10, 10, 10, 10)); 
            sidePanel.add(Box.createVerticalStrut(30));
            sidePanel.add(exhaustButton);

            // Personalizing Exhaust Plot Button
            JButton pretempButton = new JButton("<html><center>Temperature Analysis T(x)<br>and<br>Pressure Analysis P(x)</center></html>");
            pretempButton.setFont(new Font("Arial", Font.PLAIN, 18));
            pretempButton.setAlignmentX(Component.CENTER_ALIGNMENT);
            pretempButton.setMaximumSize(new Dimension(250, 80));
            pretempButton.setFocusPainted(false);
            pretempButton.setBackground(new Color(220, 220, 220));
            pretempButton.setMargin(new Insets(10, 10, 10, 10)); 
            sidePanel.add(Box.createVerticalStrut(30));
            sidePanel.add(pretempButton);

            //Adding Final Button (Exit Button)
            sidePanel.add(Box.createVerticalStrut(30));
            sidePanel.add(exitButton);

            // Button Actions
            loadParamsButton.addActionListener(e -> {
                try {
                    File csvFile = new File("nozzle_geometry.csv");
                    if (csvFile.exists()) {
                        Desktop.getDesktop().open(csvFile);
                        JOptionPane.showMessageDialog(frame, "Loading Complete!")
            ;
                    } else {
                        JOptionPane.showMessageDialog(frame, "CSV File Not Found!", "Error", JOptionPane.ERROR_MESSAGE);
                    }
                    
                } catch (IOException ex) {
                    ex.printStackTrace();
                    JOptionPane.showMessageDialog(frame, "Error Opening CSV File!", "Error", JOptionPane.ERROR_MESSAGE);
                }
            });

            runButton.addActionListener(e -> {
                try {
                    ProcessBuilder pb = new ProcessBuilder("python", "NozzleGeometry.py");
                    pb.directory(new File(System.getProperty("user.dir")));
                    pb.redirectErrorStream(true);
                    Process process = pb.start();

                    BufferedReader reader = new BufferedReader(
                        new InputStreamReader(process.getInputStream()));
                    String line;
                    while ((line = reader.readLine()) != null) {
                        System.out.println(line);
                    }

                    process.waitFor();
                    JOptionPane.showMessageDialog(frame, "Simulation Complete!");
                } catch (IOException | InterruptedException ex) {
                    ex.printStackTrace();
                    JOptionPane.showMessageDialog(frame,
                        "Error Executing Python Script:\n" + ex.getMessage(),
                        "Error", JOptionPane.ERROR_MESSAGE);
                }
            });

            throatButton.addActionListener(e -> {
                try {
                    ProcessBuilder pb = new ProcessBuilder(
                        "C:\\Users\\rafae\\AppData\\Local\\Microsoft\\WindowsApps\\python3.exe",
                        "getThroatRadius.py"
                    );

                    pb.directory(new File(System.getProperty("user.dir")));
                    pb.redirectErrorStream(true);
                    Process process = pb.start();

                    BufferedReader reader = new BufferedReader(
                        new InputStreamReader(process.getInputStream()));
                    String line;
                    while ((line = reader.readLine()) != null) {
                        System.out.println(line);
                    }

                    process.waitFor();
                    JOptionPane.showMessageDialog(frame, "Operation Complete!");
                } catch (IOException | InterruptedException ex) {
                    ex.printStackTrace();
                    JOptionPane.showMessageDialog(frame,
                        "Error Executing Python Script:\n" + ex.getMessage(),
                        "Error", JOptionPane.ERROR_MESSAGE);
                }
            });

            analysisButton.addActionListener(e -> {
                try {
                    ProcessBuilder pb = new ProcessBuilder(
                        "C:\\Users\\rafae\\AppData\\Local\\Microsoft\\WindowsApps\\python3.exe",
                        "IdealExpansionRatio.py"
                    );

                    pb.directory(new File(System.getProperty("user.dir")));
                    pb.redirectErrorStream(true);
                    Process process = pb.start();

                    BufferedReader reader = new BufferedReader(
                        new InputStreamReader(process.getInputStream()));
                    String line;
                    while ((line = reader.readLine()) != null) {
                        System.out.println(line);
                    }

                    process.waitFor();
                    JOptionPane.showMessageDialog(frame, "Operation Complete!");
                } catch (IOException | InterruptedException ex) {
                    ex.printStackTrace();
                    JOptionPane.showMessageDialog(frame,
                        "Error Executing Python Script:\n" + ex.getMessage(),
                        "Error", JOptionPane.ERROR_MESSAGE);
                }
            });

            exhaustButton.addActionListener(e -> {
                try {
                    ProcessBuilder pb = new ProcessBuilder(
                        "C:\\Users\\rafae\\AppData\\Local\\Microsoft\\WindowsApps\\python3.exe",
                        "PlotExhaustParameters.py"
                    );

                    pb.directory(new File(System.getProperty("user.dir")));
                    pb.redirectErrorStream(true);
                    Process process = pb.start();

                    BufferedReader reader = new BufferedReader(
                        new InputStreamReader(process.getInputStream()));
                    String line;
                    while ((line = reader.readLine()) != null) {
                        System.out.println(line);
                    }

                    process.waitFor();
                    JOptionPane.showMessageDialog(frame, "Operation Complete!");
                } catch (IOException | InterruptedException ex) {
                    ex.printStackTrace();
                    JOptionPane.showMessageDialog(frame,
                        "Error Executing Python Script:\n" + ex.getMessage(),
                        "Error", JOptionPane.ERROR_MESSAGE);
                }
            });

            thrustButton.addActionListener(e -> {
                try {
                    ProcessBuilder pb = new ProcessBuilder(
                        "C:\\Users\\rafae\\AppData\\Local\\Microsoft\\WindowsApps\\python3.exe",
                        "PlotThrustAnalysis.py"
                    );

                    pb.directory(new File(System.getProperty("user.dir")));
                    pb.redirectErrorStream(true);
                    Process process = pb.start();

                    BufferedReader reader = new BufferedReader(
                        new InputStreamReader(process.getInputStream()));
                    String line;
                    while ((line = reader.readLine()) != null) {
                        System.out.println(line);
                    }

                    process.waitFor();
                    JOptionPane.showMessageDialog(frame, "Operation Complete!");
                } catch (IOException | InterruptedException ex) {
                    ex.printStackTrace();
                    JOptionPane.showMessageDialog(frame,
                        "Error Executing Python Script:\n" + ex.getMessage(),
                        "Error", JOptionPane.ERROR_MESSAGE);
                }
            });

            performanceButton.addActionListener(e -> {
                try {
                    ProcessBuilder pb = new ProcessBuilder(
                        "C:\\Users\\rafae\\AppData\\Local\\Microsoft\\WindowsApps\\python3.exe",
                        "PerformanceParameters.py"
                    );

                    pb.directory(new File(System.getProperty("user.dir")));
                    pb.redirectErrorStream(true);
                    Process process = pb.start();

                    BufferedReader reader = new BufferedReader(
                        new InputStreamReader(process.getInputStream()));
                    String line;
                    while ((line = reader.readLine()) != null) {
                        System.out.println(line);
                    }

                    process.waitFor();
                    JOptionPane.showMessageDialog(frame, "Operation Complete!");
                } catch (IOException | InterruptedException ex) {
                    ex.printStackTrace();
                    JOptionPane.showMessageDialog(frame,
                        "Error Executing Python Script:\n" + ex.getMessage(),
                        "Error", JOptionPane.ERROR_MESSAGE);
                }
            });

            pretempButton.addActionListener(e -> {
                try {
                    ProcessBuilder pb = new ProcessBuilder(
                        "C:\\Users\\rafae\\AppData\\Local\\Microsoft\\WindowsApps\\python3.exe",
                        "PressureTemperatureAnalysis.py"
                    );

                    pb.directory(new File(System.getProperty("user.dir")));
                    pb.redirectErrorStream(true);
                    Process process = pb.start();

                    BufferedReader reader = new BufferedReader(
                        new InputStreamReader(process.getInputStream()));
                    String line;
                    while ((line = reader.readLine()) != null) {
                        System.out.println(line);
                    }

                    process.waitFor();
                    JOptionPane.showMessageDialog(frame, "Operation Complete!");
                } catch (IOException | InterruptedException ex) {
                    ex.printStackTrace();
                    JOptionPane.showMessageDialog(frame,
                        "Error Executing Python Script:\n" + ex.getMessage(),
                        "Error", JOptionPane.ERROR_MESSAGE);
                }
            });


            exitButton.addActionListener(e -> System.exit(0));

            frame.add(sidePanel, BorderLayout.WEST);
            frame.setVisible(true);
        });
    }
}



/*Improvments: Performance parameters (c*, Isp, Th,cf, nozzle efficiency)
 * FLuid properties (viscosity, R, gamma, Reinolds with prandlt)
 */